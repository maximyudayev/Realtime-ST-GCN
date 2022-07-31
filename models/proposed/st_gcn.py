import torch
import torch.nn as nn
import torch.nn.functional as F
from models.proposed.utils.graph import Graph

class Stgcn(nn.Module):
    """Spatial temporal graph convolutional network of Yan, et al. (2018), adapted for realtime.
    (https://arxiv.org/abs/1801.07455).

    Implements both, realtime and buffered realtime logic in the same source. 
    At runtime, the model looks at the L-dimension of the tensor to make the predictions.
    This enforces computations are numerically correct if the frame buffer is not completely full
    (e.g. the last minibatch of frames from a recording if ``capture_length % buffer != 0``).
    
    All arguments are positional to enforce separation of concern and pass the responsibility for
    model configuration up in the chain to the envoking program (config file).

    TODO:
        ``1.`` add logic for variation in FIFO latency.

    Shape:
        - Input[0]:    :math:`(N, C_{in}, L, V)`.
        - Output[0]:   :math:`(N, C_{out}, L)`. 
        
        where
            :math:`N` is a batch size.

            :math:`C_{in}` is the number of input channels (features).

            :math:`C_{out}` is the number of classification classes.

            :math:`L` is the number of frames (capture length).

            :math:`V` is the number of graph nodes.
    """

    def __init__(
        self,
        # in_feat: int,
        # num_classes: int,
        # kernel: list[int],
        # importance: bool,
        # latency: bool,
        # layers: list[int],
        # in_ch: list[list[int]],
        # out_ch: list[list[int]],
        # stride: list[list[int]],
        # residual: list[list[int]],
        # dropout: list[list[float]],
        # graph: dict,
        # strategy: str,
        **kwargs) -> None:
        """
        Kwargs:
            in_feat : ``int`` 
                Number of input sample channels/features.
            
            num_classes : ``int``
                Number of output classification classes.
            
            kernel : ``list[int]``
                Temporal kernel size Gamma.
            
            importance : ``bool``
                If ``True``, adds a learnable importance weighting to the edges of the graph.
            
            latency : ``bool``
                If ``True``, residual connection adds 
                ``x_{t}`` frame to ``y_{t}`` (which adds ``ceil(kernel_size/2)`` latency), 
                adds ``x_{t}`` frame to ``y_{t-ceil(kernel_size/2)}`` otherwise.
            
            layers : ``list[int]``
                Array of number of ST-GCN layers, oner per stage.

            in_ch : ``list[list[int]]``
                2D array of input channel numbers, one per stage per ST-GCN layer.

            out_ch : ``list[list[int]]``
                2D array of output channel numbers, one per stage per ST-GCN layer.

            stride : ``list[list[int]]``
                2D array of temporal stride sizes, one per stage per ST-GCN layer.

            residual : ``list[list[int]]``
                2D array of residual connection flags, one per stage per ST-GCN layer.

            dropout : ``list[list[float]]``
                2D array of dropout parameters, one per stage per ST-GCN layer.

            graph : ``dict`` 
                Dictionary with parameters for skeleton Graph.

            strategy : ``str``
                Type of Graph partitioning strategy.
        """

        super().__init__()
        
        # save the config arguments for model conversions
        self.conf = kwargs

        # verify that parameter dimensions match (correct number of layers/parameters per stage)
        for i, layers_in_stage in enumerate(kwargs['layers']):
            assert((len(kwargs['in_ch'][i]) == layers_in_stage) and
                    (len(kwargs['out_ch'][i]) == layers_in_stage) and
                    (len(kwargs['stride'][i]) == layers_in_stage) and
                    (len(kwargs['residual'][i]) == layers_in_stage),
                ("Incorrect number of constructor parameters in the ST-GCN stage ModuleList.\n"
                "Expected for stage {0}: {1}, got: ({2}, {3}, {4}, {5})")
                .format(
                    i, 
                    kwargs['layers'][i], 
                    len(kwargs['in_ch'][i]), 
                    len(kwargs['out_ch'][i]), 
                    len(kwargs['stride'][i]), 
                    len(kwargs['residual'][i])))

        # register the normalized adjacency matrix as a non-learnable saveable parameter 
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # input capture normalization
        # (N,C,L,V)
        self.bn_in = nn.BatchNorm2d(kwargs['in_feat'])
        
        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=kwargs['in_feat'], out_channels=kwargs['in_ch'][0][0], kernel_size=1)
        
        # stack of ST-GCN layers
        stack = [[StgcnLayer(
                    num_joints=kwargs['graph']['num_node'],
                    in_channels=kwargs['in_ch'][i][j],
                    out_channels=kwargs['out_ch'][i][j],
                    kernel_size=kwargs['kernel'][i],
                    stride=kwargs['stride'][i][j],
                    num_partitions=self.A.shape[0],
                    residual=not not kwargs['residual'][i][j],
                    dropout=kwargs['dropout'][i][j],
                    capture_length=kwargs['capture_length'])
                for j in range(layers_in_stage)] 
                for i, layers_in_stage in enumerate(kwargs['layers'])]
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        self.st_gcn = nn.ModuleList([module for sublist in stack for module in sublist])
        
        # global pooling
        # converts (N,C,L,V) -> (N,C,L,1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, kwargs['graph']['num_node']))

        # fcn for prediction
        # maps C to num_classes channels: (N,C,L,1) -> (N,F,L,1) 
        self.fcn_out = nn.Conv2d(
            in_channels=kwargs['out_ch'][-1][-1], 
            out_channels=kwargs['num_classes'], 
            kernel_size=1)

        # learnable edge importance weighting matrices (each layer, separate weighting)
        if kwargs['importance']:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(
                    torch.ones(kwargs['graph']['num_node'], 
                    kwargs['graph']['num_node'], 
                    requires_grad=True)) for _ in self.st_gcn])
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

        # removes the last dimension (node dimension) of size 1: (N,C,L,1) -> (N,C,L)
        return x.squeeze(-1)


    # def _swap_layers_for_inference(self: nn.Module) -> nn.Module:
        
    #     return


    # def train(self: nn.Module, mode: bool = True) -> nn.Module:
    #     # TODO: 
    #     return super().train(mode)

    
    def eval(self: nn.Module) -> nn.Module:
        super().eval()

        # stack of ST-GCN layers
        stack = [[RtStgcnLayer(
                    num_joints=self.conf['graph']['num_node'],
                    fifo_latency=self.conf['latency'],
                    in_channels=self.conf['in_ch'][i][j],
                    out_channels=self.conf['out_ch'][i][j],
                    kernel_size=self.conf['kernel'][i],
                    stride=self.conf['stride'][i][j],
                    num_partitions=self.A.shape[0],
                    residual=not not self.conf['residual'][i][j],
                    dropout=self.conf['dropout'][i][j],
                    **self.conf)
                for j in range(layers_in_stage)] 
                for i, layers_in_stage in enumerate(self.conf['layers'])]
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        self.st_gcn = nn.ModuleList([module for sublist in stack for module in sublist])

        # TODO: copy trained weights over from batch training to the inference layers

        return self

    
class RtStgcnLayer(nn.Module):
    """[!Inference only!] Applies a spatial temporal graph convolution over an input graph sequence.
    
    Each layer has a FIFO to store the corresponding Gamma-sized window of graph frames.
    All arguments are positional to enforce separation of concern and pass the responsibility for
    model configuration up in the chain to the envoking program (config file).

    TODO:
        ``1.`` add logic for variation in FIFO latency.

        ``2.`` write more elaborate class description about the design choices and working principle.

    Shape:
        - Input[0]:     :math:`(N, C_{in}, L, V)` - Input graph frame.
        - Input[1]:     :math:`(P, V, V)` - Graph adjacency matrix.
        - Output[0]:    :math:`(N, C_{out}, L, V)` - Output graph frame.

        where
            :math:`N` is the batch size.
            
            :math:`L` is the buffer size (buffered frames number).

            :math:`C_{in}` is the number of input channels (features).

            :math:`C_{out}` is the number of output channels (features).

            :math:`V` is the number of graph nodes.

            :math:`P` is the number of graph partitions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_joints: int,
        stride: int,
        num_partitions: int,
        dropout: float,
        residual: bool,
        fifo_latency: bool,
        **kwargs):
        """
        Args:
            in_channels : ``int``
                Number of input sample channels/features.
            
            out_channels : ``int``
                Number of channels produced by the convolution.
            
            kernel_size : ``int``
                Size of the temporal window Gamma.
            
            num_joints : ``int``
                Number of joint nodes in the graph.
            
            stride : ``int``
                Stride of the temporal reduction.
            
            num_partitions : ``int``
                Number of partitions in selected strategy.
                Must correspond to the first dimension of the adjacency tensor.
            
            dropout : ``float``
                Dropout rate of the final output.
            
            residual : ``bool``
                If ``True``, applies a residual connection.
            
            fifo_latency : ``bool``
                If ``True``, residual connection adds ``x_{t}`` frame to ``y_{t}`` (which adds ``ceil(kernel_size/2)`` latency), 
                otherwise adds ``x_{t}`` frame to ``y_{t-ceil(kernel_size/2)}``.
        """
        
        super().__init__()

        # temporal kernel Gamma is symmetric (odd number)
        # assert len(kernel_size) == 1
        assert kernel_size % 2 == 1

        self.num_partitions = num_partitions
        self.num_joints = num_joints
        self.stride = stride

        self.out_channels = out_channels
        self.fifo_size = stride*(kernel_size-1)+1
        
        # convolution of incoming frame 
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # FIFO for intermediate Gamma graph frames after multiplication with adjacency matrices
        # (N,G,P,C,V) - (N)batch, (G)amma, (P)artition, (C)hannels, (V)ertices
        self.fifo = torch.zeros(kwargs['batch_size'], self.fifo_size, num_partitions, out_channels, num_joints)
        
        # normalization and dropout on main branch
        self.bn_do = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels))

        # activation of branch sum
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        """
        In case of buffered realtime processing, Conv2D and MMUL are done on the buffered frames,
        which mimics the kernels reuse mechanism that would be followed in hardware at the expense
        of extra memory for storing intermediate results.

        TODO:
            ``1.`` Speed up the for-loop part (buffered realtime setup) by vectorization.
        """
        # residual branch
        res = self.residual(x)
        
        # spatial convolution of incoming frame (node-wise)
        a = self.conv(x)

        # convert to the expected dimension order and add the partition dimension
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        b = torch.split(a, self.out_channels, dim=1)
        # concatenate these 4D tensors across the partition dimension
        c = torch.stack(b, -1)
        # change the dimension order for the correct broadcating of the adjacency matrix
        # (N,C,L,V,P) -> (N,L,P,C,V)
        d = torch.permute(c, (0,2,4,1,3))
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        e = torch.matmul(d, A)

        # perform temporal accumulation for each of the buffered frames
        # (portability for buffered_realtime setup, for realtime, the buffer is of size 1)
        outputs = []
        for i in range(e.shape[1]):
            # push the frame into the FIFO
            self.fifo = torch.cat((e[:,i:i+1], self.fifo[:,:self.fifo_size-1]), 1)
            
            # slice the tensor according to the temporal stride size
            # (if stride is 1, returns the whole tensor itself)
            f = self.fifo[:,range(0, self.fifo_size, self.stride)]

            # sum temporally and across partitions
            # (C,H)
            g = torch.sum(f, dim=(1,2))
            outputs.append(g)

        # stack frame-wise tensors into the original length L
        # [(N,C,V)] -> (N,C,L,V)
        h = torch.stack(outputs, 2)

        # add the branches (main + residual)
        i = h + res

        return self.relu(i)


class StgcnLayer(nn.Module):
    """[Training] Applies a spatial temporal graph convolution over an input graph sequence.
    
    Processes the entire video capture during training; it is mandatory to retain intermediate values
    for backpropagation (hence no FIFOs allowed in training). Results of training with either layer
    are identical, it is simply a nuissance of autodiff frameworks.
    All arguments are positional to enforce separation of concern and pass the responsibility for
    model configuration up in the chain to the envoking program (config file).

    TODO:
        ``1.`` validate documentation.

    Shape:
        - Input[0]:     :math:`(N, C_{in}, L, V)` - Input graph frame.
        - Input[1]:     :math:`(P, V, V)` - Graph adjacency matrix.
        - Output[0]:    :math:`(N, C_{out}, L, V)` - Output graph frame.

        where
            :math:`N` is the batch size.

            :math:`C_{in}` is the number of input channels (features).

            :math:`C_{out}` is the number of output channels (features).

            :math:`L` is the video capture length.

            :math:`V` is the number of graph nodes.

            :math:`P` is the number of graph partitions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_joints: int,
        stride: int,
        num_partitions: int,
        dropout: float,
        residual: bool,
        capture_length: int):
        """
        Args:
            in_channels : ``int``
                Number of input sample channels/features.
            
            out_channels : ``int``
                Number of channels produced by the convolution.
            
            kernel_size : ``int``
                Size of the temporal window Gamma.
            
            num_joints : ``int``
                Number of joint nodes in the graph.
            
            stride : ``int``
                Stride of the temporal reduction.
            
            num_partitions : ``int``
                Number of partitions in selected strategy.
                Must correspond to the first dimension of the adjacency tensor.
            
            dropout : ``float``
                Dropout rate of the final output.
            
            residual : ``bool``
                If ``True``, applies a residual connection.
        """
        
        super().__init__()

        # temporal kernel Gamma is symmetric (odd number)
        # assert len(kernel_size) == 1
        assert kernel_size % 2 == 1

        self.num_partitions = num_partitions
        self.num_joints = num_joints
        self.stride = stride

        self.out_channels = out_channels
        # start padding across the batch dimension, accounting for strided temporal accumulation
        self.padding_size = stride*(kernel_size-1)
        self.window_size = stride*(kernel_size-1)+1

        # convolution of incoming frame 
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)
        
        # normalization and dropout on main branch
        self.bn_do = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels))

        # activation of branch sum
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, A):
        # residual branch
        res = self.residual(x)
        
        # spatial convolution of incoming frame (node-wise)
        a = self.conv(x)

        # convert to the expected dimension order and add the partition dimension
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        b = torch.split(a, self.out_channels, dim=1)
        # concatenate these 4D tensors across the partition dimension
        c = torch.stack(b, -1)
        # change the dimension order for the correct broadcating of the adjacency matrix
        # (N,C,L,V,P) -> (N,L,P,C,V)
        d = torch.permute(c, (0,2,4,1,3))
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        e = torch.matmul(d, A)
        
        # zero-pad the beginning of the sequence (L) for G-sized window to slide over, similar to a FIFO filling up
        f = F.pad(e, [0]*6+[self.padding_size,0])
        # sum temporally, don't forget to use the L-size before padding to produce an equ-length output
        outputs = []
        for id in range(e.size()[1]):
            output = torch.sum(f[:,range(id, id+self.window_size, self.stride)], dim=(1))
            outputs.append(output)

        # [(N,P,C,V)] -> (N,P,C,L,V)
        g = torch.stack(outputs, dim=3)
        # sum across partitions (N,C,L,V)
        h = torch.sum(g, dim=(1))

        # add the branches (main + residual)
        i = h + res

        return self.relu(i)
