import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ConvTemporalGraphical, Graph, LayerNorm, BatchNorm1d


class Model(nn.Module):
    """Original classification spatial temporal graph convolutional networks.

    Data provision (batching, unfolding, etc.) is delegated to the caller. Model operates 
    on frame-by-frame basis and only requires an input buffer supplied to in the size of
    the requested receptive field.

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
        super(Model, self).__init__()

        conf = kwargs['st-gcn']

        # load graph
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = kwargs['graph']['num_node']
        temporal_kernel_size = conf['kernel']
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.norm_in = LayerNorm([kwargs['in_feat'], 1, A.size(1)], device=0) if kwargs['normalization'] == 'LayerNorm' else BatchNorm1d(kwargs['in_feat'] * A.size(1), track_running_stats=False, device=0)

        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=conf['in_feat'], out_channels=conf['in_ch'][0], kernel_size=1, device=0)

        self.gcn_networks = nn.ModuleList([
            StgcnLayer(
                in_channels=conf['in_ch'][i],
                out_channels=conf['out_ch'][i],
                kernel_size=kernel_size,
                partitions=A.size(0),
                num_joints=A.size(1),
                stride=conf['stride'][i],
                residual=not not conf['residual'][i],
                dropout=conf['dropout'][i],
                normalization=kwargs['normalization'],
                device=0 if i < 5 else 1)
            for i in range(conf['layers'])])

        # initialize parameters for edge importance weighting
        if conf['importance']:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size(), device=0 if i < 5 else 1))
                for i in self.gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_networks)

        # fcn for prediction
        self.fcn_out = nn.Conv2d(
            conf['out_ch'][-1],
            out_channels=kwargs['num_classes'],
            kernel_size=1,
            device=1)


    def forward(self, x):
        self.A.to(0)
        
        # data normalization
        x = self.norm_in(x)

        # remap the features to the network size
        x = self.fcn_in(x)

        # forward
        for i, (gcn, importance) in enumerate(zip(self.gcn_networks, self.edge_importance)):
            if i == 5:
                x.to(1)
                self.A.to(1)
            x = gcn(x, self.A * importance)

        # global pooling (across time L, and nodes V)
        x = F.avg_pool2d(x, x.size()[2:])

        # prediction
        x = self.fcn_out(x)

        return x


class StgcnLayer(nn.Module):
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
        num_joints,
        stride=1,
        dropout=0,
        residual=True,
        normalization='LayerNorm',
        device=None):

        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = (((kernel_size[0] - 1) // 2), 0)

        self.gcn = ConvTemporalGraphical(
            in_channels,
            out_channels,
            kernel_size[1],
            partitions,
            device=device)

        self.tcn = nn.Sequential(
            LayerNorm([out_channels, 1, num_joints], device=device) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                stride=(stride, 1),
                padding=padding,
                device=device),
            LayerNorm([out_channels, 1, num_joints], device=device) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False, device=device),
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
                    kernel_size=1,
                    stride=(stride, 1),
                    device=device),
                LayerNorm([out_channels, 1, num_joints], device=device) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False, device=device))

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        res = self.residual(x)
        # graph convolution
        x = self.gcn(x, A)
        # temporal accumulation (but using a learnable kernel)
        x = self.tcn(x)

        return self.relu(x + res)
