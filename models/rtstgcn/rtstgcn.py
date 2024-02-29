import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F
from models.utils import Graph, LayerNorm
from torch.ao.quantization import QuantStub, DeQuantStub


class Model(nn.Module):
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
        rank=None,
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

        super(Model, self).__init__()

        # save the config arguments for model conversions
        self.conf = kwargs['rt-st-gcn']

        # register the normalized adjacency matrix as a non-learnable saveable parameter in the top-level container
        # TODO: check if the buffer gets automatically quantized (relevant only for the size estimate)
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        A = torch.tensor(self.graph.A, dtype=torch.float32, device=rank, requires_grad=False)
        self.register_buffer('A', A)

        # input capture normalization
        # (N,C,L,V)
        self.norm_in = LayerNorm([kwargs['in_feat'], 1, A.size(1)])

        # fcn for feature remapping of input to the network size
        self.fcn_in = nn.Conv2d(in_channels=self.conf['in_feat'], out_channels=self.conf['in_ch'][0], kernel_size=1)

        # stack of ST-GCN layers
        stack = [
            OfflineLayer(
                num_joints=kwargs['graph']['num_node'],
                in_channels=self.conf['in_ch'][i],
                out_channels=self.conf['out_ch'][i],
                kernel_size=self.conf['kernel'],
                stride=self.conf['stride'][i],
                num_partitions=self.A.shape[0],
                residual=not not self.conf['residual'][i],
                dropout=self.conf['dropout'][i],
                importance=self.conf['importance'],
                graph=self.A,
                segment=kwargs["segment"],
                rank=rank)
            for i in range(self.conf['layers'])]
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        self.st_gcn = nn.Sequential(*stack)

        # global pooling
        # converts (N,C,L,V) -> (N,C,L,1)
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, kwargs['graph']['num_node']))

        # fcn for prediction
        # maps C to num_classes channels: (N,C,L,1) -> (N,F,L,1)
        self.fcn_out = nn.Conv2d(
            in_channels=self.conf['out_ch'][-1],
            out_channels=kwargs['num_classes'],
            kernel_size=1)

        self.quant = QuantStub()
        self.dequant = DeQuantStub()


    def forward(self, x):
        # data normalization
        x = self.norm_in(x)

        # remap the features to the network size
        x = self.fcn_in(x)

        # feed the frame into the ST-GCN block backbone
        x = self.st_gcn(x)

        # pool the output frame for a single feature vector
        x = self.avg_pool(x)

        # remap the feature vector to class predictions
        x = self.fcn_out(x)

       # removes the last dimension (node dimension) of size 1: (N,C,L,1) -> (N,C,L)
        x = x.squeeze(-1)

        return x


    def _swap_layers_for_inference(self):
        # stack of ST-GCN layers
        stack = [
            OnlineLayer(
                num_joints=self.A.shape[-1],
                fifo_latency=self.conf['latency'],
                in_channels=self.conf['in_ch'][i],
                out_channels=self.conf['out_ch'][i],
                kernel_size=self.conf['kernel'],
                stride=self.conf['stride'][i],
                num_partitions=self.A.shape[0],
                residual=not not self.conf['residual'][i],
                dropout=self.conf['dropout'][i],
                importance=self.conf['importance'],
                graph=self.A)
            for i in range(self.conf['layers'])]
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        new_st_gcn = nn.Sequential(*stack)

        # copy parameters from the batch trained model into the RT version
        # user must ensure that all necessary updates are made on conversion between the two different architectures
        new_st_gcn.load_state_dict(self.st_gcn.state_dict(), strict=False)

        # replace the st gcn stack in the model
        self.st_gcn = new_st_gcn

        return


    def eval_(self):
        for module in self.st_gcn: module.eval_()
        return
    
    
    def prepare_dict():
        return {
            "float_to_observed_custom_module_class": {
                "static": {
                    AggregateStgcn: ObservedAggregateStgcn,
                }
            }
        }
    

    def convert_dict():
        return {
            "observed_to_quantized_custom_module_class": {
                "static": {
                    ObservedAggregateStgcn: QAggregateStgcn,
                }
            }
        }


class OfflineLayer(nn.Module):
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
        in_channels,
        out_channels,
        kernel_size,
        num_joints,
        stride,
        num_partitions,
        dropout,
        residual,
        importance,
        graph,
        segment,
        rank):
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

        super(OfflineLayer, self).__init__()

        # temporal kernel Gamma is symmetric (odd number)
        # assert len(kernel_size) == 1
        assert kernel_size % 2 == 1

        self.num_partitions = num_partitions
        self.num_joints = num_joints
        self.stride = stride
        self.kernel_size = kernel_size

        self.out_channels = out_channels

        # each layer has copy of the adjacency matrix to comply with single-input forward signature of layers for the quantization flow
        self.A = graph.clone().detach()

        # learnable edge importance weighting matrices (each layer, separate weighting)
        self.edge_importance = nn.Parameter(torch.ones(num_partitions,num_joints,num_joints,device=rank), requires_grad=True) if importance else 1

        # TODO: replace with unfold -> fold calls
        # Toeplitz matrix for temporal accumulation that mimics FIFO behavior, but in batch on full sequence
        self.toeplitz = torch.zeros(segment, segment, device=rank)
        for i in range(self.kernel_size//self.stride):
            self.toeplitz += F.pad(
                torch.eye(
                    segment - self.stride * i,
                    device=rank),
                (i*self.stride,0,0,i*self.stride))

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # normalization and dropout on main branch
        self.bn_relu = nn.Sequential(
            LayerNorm([out_channels, 1, num_joints]),
            nn.ReLU())

        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                LayerNorm([out_channels, 1, num_joints]))

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = nn.Dropout(dropout)
        else:
            self.do = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout))


    def forward(self, x):
        # residual branch
        res = self.residual(x)

        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)

        # convert to the expected dimension order and add the partition dimension 
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension) 
        # split into separate 4D tensors, each corresponding to a separate partition 
        x = torch.split(x, self.out_channels, dim=1)
        # concatenate these 4D tensors across the partition dimension
        x = torch.stack(x, -1)
        # change the dimension order for the correct broadcating of the adjacency matrix 
        # (N,C,L,V,P) -> (N,L,P,C,V)
        x = x.permute(0,2,4,1,3)
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions) 
        x = torch.matmul(x, self.A*self.edge_importance)

        # sum temporally by multiplying features with the Toeplitz matrix
        # reorder dimensions for correct broadcasted multiplication (N,L,P,C,V) -> (N,P,C,V,L) 
        x = x.permute(0,2,3,4,1)
        x = torch.matmul(x, self.toeplitz)
        # sum across partitions (N,C,V,L)
        x = torch.sum(x, dim=(1))
        # match the dimension ordering of the input (N,C,V,L) -> (N,C,L,V)
        x = x.permute(0,1,3,2)

        # normalize the output of the st-gcn operation and activate
        x = self.bn_relu(x)

        # add the branches (main + residual), activate and dropout
        return self.do(x + res)


class OnlineLayer(nn.Module):
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
        in_channels,
        out_channels,
        kernel_size,
        num_joints,
        stride,
        num_partitions,
        dropout,
        residual,
        importance,
        graph,
        segment,
        rank):
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

        super(OnlineLayer, self).__init__()

        # temporal kernel Gamma is symmetric (odd number)
        # assert len(kernel_size) == 1
        assert kernel_size % 2 == 1

        # self.num_partitions = num_partitions
        # self.num_joints = num_joints

        fifo_size = stride*(kernel_size-1)+1
        self.is_residual = residual
        self.is_residual_noconv = residual and (in_channels == out_channels) and (stride == 1)

        # learnable edge importance weighting matrices (each layer, separate weighting)
        # just a placeholder for successful load_state_dict() from <StgcnLayer>._swap_layers_for_inference()
        self.edge_importance = nn.Parameter(torch.ones(num_partitions,num_joints,num_joints), requires_grad=False) if importance else 1

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        # self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # non-traceable natively spatial and temporal aggregation of remapped node features
        # split into a separate module for the quantizaiton workflow
        self.aggregate = AggregateStgcn(graph, fifo_size, kernel_size, out_channels, stride)

        # normalization and dropout on main branch
        self.bn_relu = nn.Sequential(
            LayerNorm([out_channels, 1, num_joints]),
            nn.ReLU())

        # residual branch
        if residual and not ((in_channels == out_channels) and (stride == 1)):
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                LayerNorm([out_channels, 1, num_joints]),
            )
        else:
            self.residual = nn.Identity()

        # functional quantizeable module for the addition of branches
        self.functional_add = nnq.FloatFunctional()
        self.functional_mul_zero = nnq.FloatFunctional()

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = nn.Dropout(dropout)
        else:
            self.do = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout))


    def eval_(self):
        # eval is called after OfflineLayer was switched for OnlineLayer with load_state_dict()
        self.aggregate.A *= self.edge_importance
        return


    def forward(self, x):
        """
        In case of buffered realtime processing, Conv2D and MMUL are done on the buffered frames,
        which mimics the kernels reuse mechanism that would be followed in hardware at the expense
        of extra memory for storing intermediate results.
        """

        # residual branch
        if not self.is_residual:
            res = self.functional_mul_zero.mul_scalar(x, 0.0)
        else:
            res = self.residual(x)

        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)

        # aggregate features spatially and temporally (non-traceable module for FX Graph Quantization)
        x = self.aggregate(x)

        # normalize the output of the st-gcn operation and activate
        x = self.bn_relu(x)

        # add the branches (main + residual), activate and dropout
        x = self.functional_add.add(x, res)

        return self.do(x)


class AggregateStgcn(nn.Module):
    """
    Floating-point custom Module for spatial-temporal aggregation separated from other natively-traceable modules for the quantization workflow
    """

    def __init__(
        self,
        graph,
        fifo_size,
        kernel_size,
        out_channels,
        stride):

        super(AggregateStgcn, self).__init__()

        self.out_channels = out_channels
        self.stride = stride
        self.fifo_size = fifo_size
        self.kernel_size = kernel_size
        self.fifo = torch.zeros(1, fifo_size, out_channels, graph.size(1), dtype=torch.float32)

        # FIFO for intermediate Gamma graph frames after multiplication with adjacency matrices
        # (N,G,C,V) - (N)batch, (G)amma, (C)hannels, (V)ertices
        # each layer has copy of the adjacency matrix to comply with single-input forward signature of layers for the quantization flow
        self.A = torch.tensor(graph, dtype=torch.float32, requires_grad=False)


    def forward(self, x):
        # TODO: optimize data manipulation and vectorize the for-loop for the buffered realtime setup

        # convert to the expected dimension order and add the partition dimension
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        x = torch.split(x, self.out_channels, dim=1)
        # concatenate these 4D tensors across the partition dimension
        x = torch.stack(x, -1)
        # change the dimension order for the correct broadcating of the adjacency matrix
        # (N,C,L,V,P) -> (N,L,P,C,V)
        x = x.permute(0,2,4,1,3)
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        x = torch.matmul(x, self.A)

        # perform temporal accumulation for each of the buffered frames
        # (portability for buffered_realtime setup, for realtime, the buffer is of size 1)
        outputs = []
        for i in range(x.shape[1]):
            # push the frame, summed across partitions, into the FIFO
            self.fifo = torch.cat((x.sum(dim=2)[:,i:i+1], self.fifo[:,:self.fifo_size-1]), 1)

            # slice the tensor according to the temporal stride size
            # (if stride is 1, returns the whole tensor itself)
            a = self.fifo[:,range(0, self.fifo_size, self.stride)]
            
            # sum temporally
            # (C,H)
            b = torch.sum(a, dim=(1))
            outputs.append(b)

        # stack frame-wise tensors into the original length L
        # [(N,C,V)] -> (N,C,L,V)
        return torch.stack(outputs, 2)


class ObservedAggregateStgcn(nn.Module):
    """"""

    def __init__(
        self,
        channels,
        num_partitions,
        num_joints,
        stride,
        fifo_size,
        kernel_size,
        graph):

        super(ObservedAggregateStgcn, self).__init__()

        self.conv3d_matmul = nn.Conv3d(in_channels=num_partitions, out_channels=num_joints, kernel_size=(1,num_joints,1), bias=False)
        self.conv3d_sum = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,kernel_size,1), dilation=(1,stride,1), bias=False)
        self.functional_cat = nnq.FloatFunctional()

        self.conv3d_matmul.weight = nn.Parameter(graph.permute(2,0,1)[:,:,None,:,None], requires_grad=False)
        self.conv3d_sum.weight = nn.Parameter(torch.ones(1,1,1,kernel_size,1), requires_grad=False)

        self.fifo = torch.zeros(1, 1, channels, fifo_size, num_joints, dtype=torch.float32)

        self.channels = channels
        self.fifo_size = fifo_size
        self.kernel_size = kernel_size
        self.num_joints = num_joints


    @classmethod
    def from_float(cls, float_mod):
        assert hasattr(float_mod, 'qconfig')
        observed_mod = cls(
            float_mod.out_channels,
            float_mod.A.size(0),
            float_mod.A.size(1),
            float_mod.stride,
            float_mod.fifo_size,
            float_mod.kernel_size,
            float_mod.A)
        observed_mod.qconfig = float_mod.qconfig
        # observed_mod.eval()
        return observed_mod


    def forward(self, x):
        x = x.unfold(dimension=1, size=self.channels, step=self.channels)
        x = self.conv3d_matmul(x).permute(0,2,4,3,1)

        # TODO: instead of -1 use the latency number of quants to make compatible with buffered RT
        self.fifo = self.functional_cat.cat([x, self.fifo[:,:,:,:-1]], dim=3)

        x = self.conv3d_sum(self.fifo)[:,0]

        return x


class QAggregateStgcn(nn.Module):
    """"""

    def __init__(
        self,
        conv3d_matmul,
        conv3d_sum,
        functional_cat,
        fifo,
        channels):

        super(QAggregateStgcn, self).__init__()

        self.conv3d_matmul = conv3d_matmul
        self.conv3d_sum = conv3d_sum
        self.functional_cat = functional_cat

        self.fifo = fifo

        self.channels = channels


    @classmethod
    def from_observed(cls, observed_mod):
        assert hasattr(observed_mod, 'qconfig')
        assert hasattr(observed_mod, 'activation_post_process')

        observed_mod.conv3d_matmul.activation_post_process = observed_mod.activation_post_process
        observed_mod.conv3d_sum.activation_post_process = observed_mod.activation_post_process
        observed_mod.conv3d_matmul.qconfig = observed_mod.qconfig
        observed_mod.conv3d_sum.qconfig = observed_mod.qconfig

        scale, zero_point = observed_mod.activation_post_process.calculate_qparams()
        quantized_mod = cls(
            nnq.Conv3d.from_float(observed_mod.conv3d_matmul),
            nnq.Conv3d.from_float(observed_mod.conv3d_sum),
            observed_mod.functional_cat,
            torch.quantize_per_tensor(observed_mod.fifo, scale, zero_point, dtype=torch.quint8),
            observed_mod.channels)
        return quantized_mod


    def forward(self, x):
        x = x.unfold(dimension=1, size=self.channels, step=self.channels)
        x = self.conv3d_matmul(x).permute(0,2,4,3,1)

        # TODO: instead of -1 use the latency number of quants to make compatible with buffered RT
        self.fifo = self.functional_cat.cat([x, self.fifo[:,:,:,:-1]], dim=3)

        x = self.conv3d_sum(self.fifo)[:,0]

        return x
