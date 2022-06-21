import torch
import torch.nn as nn
from utils.graph import Graph

class Model(nn.Module):
    r"""Spatial temporal graph convolutional network of Yan, et al. (2018), adapted for realtime.
    (https://arxiv.org/abs/1801.07455).

    TODO:
        ``1.`` add logic for variation in FIFO latency

        ``2.`` enable batch processing (N frames buffered and processed in batch)

        ``3.`` move all hardcoded parameters into configuration files for manageability and scalability

    Args:
        input_dim                   (int): Number of input sample channels/features
        
        num_classes                 (int): Number of output classification classes
        
        num_joints                  (int): Number of nodes/joints in the skeleton 
        
        kernel_size                 (int, optional): Temporal kernel size Gamma. Default: ``9``
        
        edge_importance_weighting   (bool, optional): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: ``True``
        
        fifo_latency                (bool, optional): If ``True``, residual connection adds 
            ``x_{t}`` frame to ``y_{t}`` (which adds ``ceil(kernel_size/2)`` latency), 
            adds ``x_{t}`` frame to ``y_{t-ceil(kernel_size/2)}`` otherwise. Default: ``False``
        
        dropout                     ([int], optional): Array of dropout parameters, one per ST-GCN layer.
            Default: ``9*[0.5]``

        graph_args                  (dict, optional): Dictionary with parameters for skeleton Graph.
            Default: ``{'layout': 'openpose', 'strategy': 'spatial'}``

    Shape:
        - Input[0]:    :math:`(N, C_{in}, V)`
        - Output[0]:   :math:`(C_{out})` 
        
        where
            :math:`N` is a batch size,
            :math:`C_{in}` is the number of input channels (features),
            :math:`C_{out}` is the number of classification classes,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, 
                input_dim, 
                num_classes, 
                num_joints,
                kernel=[9],
                edge_importance_weighting=True,
                fifo_latency=False,
                dropout=9*[0.5],
                graph_args = {'layout': 'openpose', 'strategy': 'spatial'},
                **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.graph = Graph(**graph_args)
        # adjacency matrix must be returned as (P, V, V)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        ########### TODO: remove after verification ##########
        num_partitions = 3
        assert(
            A.shape == (num_partitions, num_joints, num_joints),
            "Unexpected Adjacency tensor dimension in Model.\n" +
            f"Expected: {(num_partitions, num_joints, num_joints)}, got: {A.shape}")
        ######################################################

        # input capture normalization
        self.bn_in = nn.BatchNorm1d(input_dim)
        
        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=1)
        
        # stack of ST-GCN layers
        layers = [9]
        in_ch = [4*[64]+3*[128]+2*[256]]
        out_ch = [3*[64]+3*[128]+3*[256]]
        stride = [[1, 1, 1, 2, 1, 1, 2, 1, 1]]
        residual = [[False]+8*[True]]

        for i in layers:
            assert(len(in_ch[i]) == layers[i] &
                    len(out_ch[i]) == layers[i] &
                    len(stride[i]) == layers[i] &
                    len(residual[i]) == layers[i],
                "Incorrect number of constructor parameters in the ST-GCN stage ModuleList.\n" +
                f"Expected all: {layers[i]}, got: {(len(in_ch[i]), len(out_ch[i]), len(stride[i]), len(residual[i]))}")

        stack = [[ST_GCN(
                    in_channels=in_ch[i][j],
                    out_channels=out_ch[i][j],
                    kernel_size=kernel[i][j],
                    num_joints=num_joints,
                    stride=stride[i][j],
                    residual=residual[i][j],
                    fifo_latency=fifo_latency,
                    dropout=dropout[i][j])
                for j in range(layers[i])] for i in layers]
        self.st_gcn = nn.ModuleList(stack)
        
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

    TODO: 
        ``1.`` add logic for variation in FIFO latency

        ``2.`` validate the logic (due to tensor dimensions) on batch processing (several frames at a time)

    Args:Model()
        in_channels     (int): Number of input sample channels/features
        
        out_channels    (int): Number of channels produced by the convolution
        
        kernel_size     (int): Size of the temporal window Gamma
        
        num_joints      (int): Number of the joint nodes in the graph 
        
        stride          (int, optional): Stride of the temporal reduction. Default: ``1``
        
        num_partitions  (int, optional): Number of partitions in selected strategy. Default: ``3`` (spatial partitioning)
            must correspond to the first dimension of the adjacency matrix (matrices)
        
        dropout         (int, optional): Dropout rate of the final output. Default: ``0.5``
        
        residual        (bool, optional): If ``True``, applies a residual connection. Default: ``True``
        
        fifo_latency    (bool, optional): If ``True``, residual connection adds 
            ``x_{t}`` frame to ``y_{t}`` (which adds ``ceil(kernel_size/2)`` latency), 
            adds ``x_{t}`` frame to ``y_{t-ceil(kernel_size/2)}`` otherwise. Default: ``False``

    Shape:
        - Input[0]:     :math:`(N, C_{in}, V)` - Input graph frame
        - Input[1]:     :math:`(P, V, V)` - Graph adjacency matrix
        - Output[0]:    :math:`(N, C_{out}, V)` - Output graph frame

        where
            :math:`N` is the batch size,
            :math:`C_{in}` is the number of input channels (features),
            :math:`C_{out}` is the number of output channels (features),
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
                residual=True,
                fifo_latency=False):
        super().__init__()

        # temporal kernel Gamma is symmetric (odd number)
        assert len(kernel_size) == 1
        assert kernel_size[0] % 2 == 1

        self.num_partitions = num_partitions
        self.stide = stride
        self.out_channels = out_channels
        self.fifo_size = 2*kernel_size-1

        ########### TODO: remove after verification ##########
        # temp saved variables 
        self.num_joints = num_joints
        ######################################################
        
        # convolution of incoming frame 
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # FIFO for intermediate Gamma graph frames after multiplication with adjacency matrices
        # (G,P,C,V) - (G)amma, (P)artition, (C)hannels, (V)ertices
        self.fifo = torch.zeros(self.fifo_size, num_partitions, out_channels, num_joints)
        
        # normalization and dropout on main branch
        self.bn_do = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

        # activation of branch sum
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        ########### TODO: remove after verification ##########
        assert(
            x.shape == (1, self.num_joints, self.in_channels),
            "Unexpected input tensor dimension in ST-GCN.\n" + 
            f"Expected: {(1, self.num_joints, self.in_channels)}, got: {x.shape}")
        assert(
            A.shape == (self.num_partitions, self.num_joints, self.num_joints),
            "Unexpected Adjacency tensor dimension in ST-GCN.\n" +
            f"Expected: {(self.num_partitions, self.num_joints, self.num_joints)}, got: {A.shape}")
        ######################################################

        # residual branch
        res = self.residual(x)
        
        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)
        
        ########### TODO: remove after verification ##########
        assert(
            x.shape == (1, self.out_channels*self.num_partitions, self.num_joints, 1),
            "Unexpected tensor dimension in ST-GCN after spatial FC.\n" +
            f"Expected: {(1, self.out_channels*self.num_partitions, self.num_joints, 1)}, got: {x.shape}")
        ######################################################

        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        a = torch.split(x, self.out_channels, dim=1)
        # concatenate these 4D tensors across the batch dimension batch
        b = torch.cat(a, 0)
        # change the dimension order for the correct broadcating of the adjacency matrix
        # (P,C,V,N) -> (N,P,C,V)
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

        ########### TODO: remove after verification ##########
        assert(
            x.shape == res.shape,
            "Tensor dimension mismatch between main and residual branches.\n" +
            f"Main: {x.shape}, residual: {res.shape}")
        ######################################################

        # add the branches (main + residual)
        x = x + res

        return self.relu(x)
