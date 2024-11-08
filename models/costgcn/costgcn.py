import torch
import torch.nn as nn
import torch.nn.quantized as nnq
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
        self.dilation = conf['dilation'][-1]

        self.norm_in = LayerNorm([kwargs['in_feat'], 1, A.size(1)]) if kwargs['normalization'] == 'LayerNorm' else BatchNorm1d(kwargs['in_feat'] * A.size(1), track_running_stats=False)

        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=conf['in_feat'], out_channels=conf['in_ch'][0], kernel_size=1)

        self.gcn_networks = nn.ModuleList([
            StgcnLayer(
                in_channels=conf['in_ch'][i],
                out_channels=conf['out_ch'][i],
                kernel_size=kernel_size,
                partitions=A.size(0),
                num_joints=A.size(1),
                stride=conf['stride'][i],
                dilation=conf['dilation'][i],
                residual=not not conf['residual'][i],
                dropout=conf['dropout'][i],
                normalization=kwargs['normalization'])
            for i in range(conf['layers'])])

        # initialize parameters for edge importance weighting
        if conf['importance']:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for _ in self.gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.gcn_networks)

        # fcn for prediction
        self.fcn_out = nn.Conv2d(
            conf['out_ch'][-1],
            out_channels=kwargs['num_classes'],
            kernel_size=1)


    def forward(self, x):
        # data normalization
        x = self.norm_in(x)

        # remap the features to the network size
        x = self.fcn_in(x)

        # forward
        for gcn, importance in zip(self.gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # global pooling (across time L, and nodes V)
        x = F.avg_pool2d(x, x.size()[2:])

        # prediction
        x = self.fcn_out(x)

        return x.squeeze(-1)


    def prepare_benchmark(self, arch_conf):
        return arch_conf


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
        dilation=1,
        dropout=0,
        residual=True,
        normalization='LayerNorm'):

        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        self.gamma = kernel_size[0]
        self.stride = stride
        self.dilation = dilation
        self.fifo_size = stride*(self.gamma-1)+1

        # number of input frames, before pooling (due to strided convolution)
        self.fifo = torch.zeros(1, out_channels, self.fifo_size, num_joints, dtype=torch.float32)
        self.fifo_res = torch.zeros(1, out_channels, self.fifo_size, num_joints, dtype=torch.float32)

        self.is_residual = residual
        self.is_residual_conv = residual and not ((in_channels == out_channels) and (stride == 1))

        self.gcn = ConvTemporalGraphical(
            in_channels,
            out_channels,
            kernel_size[1],
            partitions)

        self.tcn = nn.Sequential(
            LayerNorm([out_channels, 1, num_joints]) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                dilation=(stride, 1),
                padding='valid'),
            LayerNorm([out_channels, 1, num_joints]) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.Dropout(dropout, inplace=True))

        # residual branch
        if self.is_residual_conv:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1),
                LayerNorm([out_channels, 1, num_joints]) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False))
        else:
            self.residual = nn.Identity()

        # functional quantizeable module for the addition of branches
        self.functional_add = nnq.FloatFunctional()
        self.functional_mul_zero = nnq.FloatFunctional()
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        # residual branch
        # add delay and another fifo
        if not self.is_residual:
            res = self.functional_mul_zero.mul_scalar(x, 0.0)
        else:
            res = self.residual(x)

        self.fifo_res = torch.cat((res, self.fifo_res[:,:,:-1]), dim=2)
        
        # graph convolution
        x = self.gcn(x, A)

        # push data into the buffer
        self.fifo = torch.cat((x, self.fifo[:,:,:-1]), dim=2)

        # temporal accumulation (but using a learnable kernel)
        x = self.tcn(self.fifo)

        return self.relu(self.functional_add.add(x, self.fifo_res[:,:,self.gamma//2:self.gamma//2+1]))
