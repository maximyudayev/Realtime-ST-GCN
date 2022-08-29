import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.tgcn import ConvTemporalGraphical
from models.utils.graph import Graph


class Model(nn.Module):
    """Spatial temporal graph convolutional networks.
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = kwargs['graph']['num_node']
        temporal_kernel_size = kwargs['kernel'][0]
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        self.data_bn = nn.BatchNorm1d(kwargs['in_feat'] * A.size(1))
        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=kwargs['in_feat'], out_channels=kwargs['in_ch'][0][0], kernel_size=1)

        stack = [[st_gcn(
                    in_channels=kwargs['in_ch'][i][j],
                    out_channels=kwargs['out_ch'][i][j],
                    kernel_size=kernel_size,
                    partitions=A.size(0),
                    stride=kwargs['stride'][i][j],
                    residual=not not kwargs['residual'][i][j],
                    dropout=kwargs['dropout'][i][j])
                for j in range(layers_in_stage)] 
                for i, layers_in_stage in enumerate(kwargs['layers'])]
        self.st_gcn_networks = nn.ModuleList([module for sublist in stack for module in sublist])

        # initialize parameters for edge importance weighting
        if kwargs['importance']:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(
            kwargs['out_ch'][-1][-1], 
            out_channels=kwargs['num_classes'],
            kernel_size=1)


    def forward(self, x):
        # # data normalization
        # N, C, T, V, M = x.size()
        # # permutes must copy the tensor over as contiguous because .view() needs a contiguous tensor
        # # this incures extra overhead
        # x = x.permute(0, 4, 3, 1, 2).contiguous()
        # # (N,M,V,C,T)
        # x = x.view(N * M, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, M, V, C, T)
        # x = x.permute(0, 1, 3, 4, 2).contiguous()
        # x = x.view(N * M, C, T, V)
        # # (N',C,T,V)

        # data normalization
        N, C, T, V = x.size()
        # permutes must copy the tensor over as contiguous because .view() needs a contiguous tensor
        # this incures extra overhead
        x = x.permute(0, 3, 1, 2).contiguous()
        # (N,V,C,T)
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1)
        # (N,C,T,V)

        # remap the features to the network size
        x = self.fcn_in(x)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, (1, x.size()[-1]))

        # prediction
        x = self.fcn(x)
        x = x.squeeze(-1)

        return x


    def extract_feature(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        # permutes must copy the tensor over as contiguous because .view() needs a contiguous tensor
        # this incures extra overhead
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        # (N,M,V,C,T)
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        # (N',C,T,V)

        # remap the features to the network size
        x = self.fcn_in(x)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, (1, x.size()[-1]))

        feature = x.squeeze(-1)

        # prediction
        x = self.fcn(x)
        output = x.squeeze(-1)

        return output, feature


class st_gcn(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        partitions,
        stride=1,
        dropout=0,
        residual=True):
        
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (((kernel_size[0] - 1) // 2) * stride, 0)

        self.gcn = ConvTemporalGraphical(
            in_channels, 
            out_channels,
            kernel_size[1],
            partitions)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                dilation=(stride, 1),
                padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1),
                nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        res = self.residual(x)
        # graph convolution
        x, A = self.gcn(x, A)
        # temporal accumulation (but using a learnable kernel)
        x = self.tcn(x)

        return self.relu(x + res), A
        