import torch
import torch.nn as nn
from utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional network of Yan, et al. (2018), adapted for realtime.
    (https://arxiv.org/abs/1801.07455).

    Args:
        in_channels                 (int): Number of input sample channels/features
        num_class                   (int): Number of output classification classes
        graph_args                  (dict): Arguments for building the graph
        edge_importance_weighting   (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph

    Shape:
        - Input:    :math:`(N, in_channels, V_{in})`
        - Output:   :math:`(num_class)` 
        
        where
            :math:`N` is a batch size,
            :math:`V_{in}` is the number of graph nodes.
    """

    def __init__(self, 
                in_channels, 
                num_classes, 
                num_joints,
                edge_importance_weighting):
        super().__init__()

        kernel_size = 9
        self.num_classes = num_classes
        
        graph_args = {'layout': 'tp-vicon', 'strategy': 'spatial'}
        self.graph = Graph(**graph_args)
        # adjacency matrix must be returned as (N*P, N)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # input capture normalization
        self.bn_in = nn.BatchNorm1d(in_channels)
        
        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1)
        
        # stack of ST-GCN layers
        self.st_gcn = nn.ModuleList([
            ST_GCN(64, 64, kernel_size, num_joints, 1, residual=False),
            ST_GCN(64, 64, kernel_size, num_joints, 1),
            ST_GCN(64, 64, kernel_size, num_joints, 1),
            ST_GCN(64, 128, kernel_size, num_joints, 2),
            ST_GCN(128, 128, kernel_size, num_joints, 1),
            ST_GCN(128, 128, kernel_size, num_joints, 1),
            ST_GCN(128, 256, kernel_size, num_joints, 2),
            ST_GCN(256, 256, kernel_size, num_joints, 1),
            ST_GCN(256, 256, kernel_size, num_joints, 1)
        ])

        # global pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=(num_joints, 1))

        # fcn for prediction
        self.fcn_out = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)

        # edge importance weighting matrices (each layer, separate weighting)
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(num_joints, num_joints))
                for _ in self.st_gcn
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn)

    def forward(self, x):
        # normalize the input frame
        x = self.bn_in(x)

        # remap the features to the network size
        x = self.fcn_in(x)

        # feed the frame into the ST-GCN block
        for st_gcn, importance in zip(self.st_gcn, self.edge_importance):
            # adjacency matrix is a 3D tensor (size depends on the partition strategy)
            x = st_gcn(x, torch.mul(self.A, importance))

        # pool the output frame for a single feature vector
        x = self.avg_pool(x)

        # remap the feature vector to class predictions
        x = self.fcn_out(x)

        # unroll the tensor into an array
        return x.view(self.num_classes)

class ST_GCN(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Each layer has a FIFO to store the corresponding Gamma-sized window of graph frames.

    Args:
        in_channels     (int): Number of input sample channels/features
        out_channels    (int): Number of channels produced by the convolution
        kernel_size     (int): Size of the temporal window Gamma
        num_joints      (int): Number of the joint nodes in the graph 
        stride          (int, optional): Stride of the temporal reduction. Default: 1
        num_partitions  (int, optional): Number of partitions in selected strategy. Default: 3 (spatial partitioning)
            must correspond to the first dimension of the adjacency matrix (matrices)
        dropout         (int, optional): Dropout rate of the final output. Default: 0.5
        residual        (bool, optional): If ``True``, applies a residual connection. Default: ``True``

    Shape:
        - Input[0]:     :math:`(N, in_channels, V)` - Input graph frame
        - Input[1]:     :math:`(P, V, V)` - Graph adjacency matrix
        - Output[0]:    :math:`(N, out_channels, V)` - Output graph frame

        where
            :math:`N` is a batch size,
            :math:`V` is the number of graph nodes,
            :math:`P` is the number of graph partitions.
    """

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                num_joints,
                stride=1,
                num_partitions=3,
                dropout=0.5,
                residual=True):
        super().__init__()

        # temporal kernel Gamma is symmetric (odd number)
        assert len(kernel_size) == 1
        assert kernel_size[0] % 2 == 1

        self.num_partitions = num_partitions
        self.stide = stride
        self.out_channels = out_channels
        self.fifo_size = 2*kernel_size-1
        
        # convolution of incoming frame 
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # FIFO for intermediate $\Gamma$ graph frames after multiplication with adjacency matrices
        # (G,P,C,H) - (G)amma, (P)artition, (C)hannel, (H)eight
        self.fifo = torch.zeros(self.fifo_size, num_partitions, out_channels, num_joints)
        
        # normalization and dropout on main branch
        self.bn_do = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        # residual branch
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        # activation of branch sum
        self.relu = nn.ReLU(inplace=True)

    # TODO: validate the logic (due to tensor dimensions) on batch processing (several frames at a time)
    def forward(self, x, A):
        # residual branch
        res = self.residual(x)
        
        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)
        
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        a = torch.split(x, self.out_channels, dim=1)
        # concatenate these 4D tensors across the batch dimension batch
        b = torch.cat(a, 0)
        # change the dimension order for the correct broadcating of the adjacency matrix
        # (N,C,H,W) -> (W,N,C,H)
        c = torch.permute(b, (3,0,1,2))
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        d = torch.matmul(c, A)

        # push the frame into the FIFO
        self.fifo = torch.cat((d, self.fifo[:self.fifo_size-1]), 0)

        # sum temporally and across partitions
        # (C,H)
        e = torch.sum(self.fifo, dim=(0,1))
        # reshape the tensor to the dimensions of the input frame
        x = torch.permute(e, (1,0))[None,:]

        # add the branches (main + residual)
        x = x + res

        return self.relu(x)