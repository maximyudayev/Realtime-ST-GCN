import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.functional as F
from models.utils import Graph, LayerNorm, BatchNorm1d
# from torch.ao.quantization import QuantStub, DeQuantStub
import numpy as np


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
        # print("graph below")
        # print(self.graph.__str__())
        # print(self.graph.__str__().shape)
        # print("graph ended")  
        A1 = torch.tensor(self.graph.A[0:1], dtype=torch.float32, device=rank, requires_grad=False).reshape(625)
        A2 = torch.tensor(self.graph.A[1:2], dtype=torch.float32, device=rank, requires_grad=False).reshape(625)
        A3 = torch.tensor(self.graph.A[2:3], dtype=torch.float32, device=rank, requires_grad=False).reshape(625)
        self.register_buffer('A1', A1)
        self.register_buffer('A2', A2)
        self.register_buffer('A3', A3)

        # stack of ST-GCN layers
        stack = [
            OfflineLayer(
                num_joints=kwargs['graph']['num_node'], #25
                in_channels=self.conf['in_ch'][i],          # 64,64,64,64,128,128,128,256,256
                out_channels=self.conf['out_ch'][i],        # 64,64,64,128,128,128,256,256,256
                kernel_size=self.conf['kernel'],            # 9
                stride=self.conf['stride'][i],              # 1,1,1,2,1,1,2,1,1      stride*(kernel_size-1)+1 
                num_partitions=3,             
                residual=not not self.conf['residual'][i],
                dropout=self.conf['dropout'][i],            # 0,0,0,0,0,0,0,0,0
                importance=self.conf['importance'],         # True
                graph1=self.A1,
                graph2=self.A2,
                graph3=self.A3,
                rank=rank,
                normalization=kwargs['normalization'])      # LayerNorm
            for i in range(self.conf['layers'])] # 9 layers
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        self.st_gcn = nn.ModuleList(stack)
        
        # global pooling
        # converts (N,C,L,V) -> (N,C,L,1)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, kwargs['graph']['num_node']))

        # fcn for prediction
        # maps C to num_classes channels: (N,C,L,1) -> (N,F,L,1)
        self.fcn_out = nn.Conv2d(
            in_channels=self.conf['out_ch'][-1],
            out_channels=kwargs['num_classes'],
            kernel_size=1)

        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()


    def forward(self, x_res, x1, x2, x3, fifo):

        # feed the frame into the ST-GCN block backbone
        for gcn in self.st_gcn:
            x = gcn(x_res, x1, x2, x3, fifo, self.A1, self.A2, self.A3)
        # x = self.st_gcn(x, self.A1, self.A2, self.A3)
        # print(x1, x2, x3)
        # print("x shape before pooling: ", x.shape)
    #     # pool the output frame for a single feature vector
        x = self.avg_pool(x)

    # #     # remap the feature vector to class predictions
            # with torch.no_grad():
        # print("shape before second fcn: ", x.shape)
        x = self.fcn_out(x)
        # print("shape after second fcn: ", x.shape)
    # #    # removes the last dimension (node dimension) of size 1: (N,C,L,1) -> (N,C,L)
        return x
        # return x[:,:,:,0]


    def _swap_layers_for_inference(self):
        # stack of ST-GCN layers
        stack = [
            OnlineLayer(
                num_joints=25,
                # fifo_latency=self.conf['latency'],
                in_channels=self.conf['in_ch'][i],
                out_channels=self.conf['out_ch'][i],
                kernel_size=self.conf['kernel'],
                stride=self.conf['stride'][i],
                num_partitions=3,
                residual=not not self.conf['residual'][i],
                dropout=self.conf['dropout'][i],
                importance=self.conf['importance'],
                graph1=self.A1,
                graph2=self.A2,
                graph3=self.A3)
            for i in range(self.conf['layers'])]
        # flatten into a single sequence of layers after parameters were used to construct
        # (done like that to make config files more readable)
        new_st_gcn = nn.Sequential(*stack)
        print("Shape in swap layers: ", self.A1.shape)
        # copy parameters from the batch trained model into the RT version
        # user must ensure that all necessary updates are made on conversion between the two different architectures
        new_st_gcn.load_state_dict(self.st_gcn.state_dict(), strict=False)

        # replace the st gcn stack in the model
        self.st_gcn = new_st_gcn

        return


    def eval_(self):
        for module in self.st_gcn: module.eval_()
        return
    


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
        graph1,
        graph2,
        graph3,
        rank,
        normalization='LayerNorm'):
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
        self.rank = rank

        self.out_channels = out_channels

        # each layer has copy of the adjacency matrix to comply with single-input forward signature of layers for the quantization flow
        self.A1 = graph1.clone().detach()
        self.A2 = graph2.clone().detach()
        self.A3 = graph3.clone().detach()
        # self.A = graph.clone().detach()

        # learnable edge importance weighting matrices (each layer, separate weighting)
        self.edge_importance = nn.Parameter(torch.ones(num_partitions,num_joints*num_joints,device=rank), requires_grad=False) if importance else 1

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)

        # normalization and dropout on main branch
        self.bn_relu = nn.Sequential(
            LayerNorm([out_channels, 1, num_joints]) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False),
            nn.ReLU())

        # residual branch
        if not residual:
            self.residual = lambda _: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                LayerNorm([out_channels, 1, num_joints]) if normalization == 'LayerNorm' else nn.BatchNorm2d(out_channels, track_running_stats=False))

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = nn.Dropout(dropout)
        else:
            self.do = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout))


    def forward(self, x, A1, A2, A3):
        _,_,L,_ = x.size()
        # residual branch
        res = self.residual(x)

        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)

        # convert to the expected dimension order and add the partition dimension 
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension) 
        # split into separate 4D tensors, each corresponding to a separate partition 
        x = torch.split(x, self.out_channels, dim=1)
        # Create a tuple from the list
        x = tuple(x) 

        print("type: ", type(x))  # This will print the type of the container holding the split tensors
        # concatenate these 4D tensors across the partition dimension
        x = torch.stack(x, -1)
        # change the dimension order for the correct broadcating of the adjacency matrix 
        # (N,C,L,V,P) -> (N,L,P,C,V)
        x = x.permute(0,2,4,1,3)
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions) 
        x = torch.matmul(x, self.A*self.edge_importance)

        # TODO: replace with unfold -> fold calls
        # Toeplitz matrix for temporal accumulation that mimics FIFO behavior, but in batch on full sequence
        toeplitz = torch.zeros(L, L, device=self.rank)
        for i in range(self.kernel_size//self.stride):
            toeplitz += F.pad(
                torch.eye(
                    L - self.stride * i,
                    device=self.rank),
                (i*self.stride,0,0,i*self.stride))
        
        # sum temporally by multiplying features with the Toeplitz matrix
        # reorder dimensions for correct broadcasted multiplication (N,L,P,C,V) -> (N,P,C,V,L) 
        x = x.permute(0,2,3,4,1)
        x = torch.matmul(x, toeplitz)
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
        graph1,
        graph2,
        graph3):
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
        print("num joints is: ", num_joints)
        self.edge_importance = nn.Parameter(torch.ones(num_partitions, num_joints* num_joints), requires_grad=False) if importance else 1

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        # self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1, bias=False)
        # self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.A1 = torch.tensor(graph1, dtype=torch.float32, requires_grad=False)
        print("shape of A1 in online layer:  ", self.A1.shape)
        self.A2 = torch.tensor(graph2, dtype=torch.float32, requires_grad=False)
        self.A3 = torch.tensor(graph3, dtype=torch.float32, requires_grad=False)

        # fifo = torch.zeros(1, fifo_size, out_channels, graph1.reshape(1,25,25).size(1), dtype=torch.float32, requires_grad=False)
        # self.register_buffer('fifo', fifo)
        
        self.fifo_size = fifo_size
        self.stride = stride
        selector2 = torch.eye(9, 8)
        self.register_buffer('selector2', selector2)
        selector = torch.eye(9, 9)
        self.register_buffer('selector', selector)

        self.conv1x1 = nn.Conv2d(9, 1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.conv1x1_weights = torch.ones_like(self.conv1x1.weight)
        self.conv1x1.weight = nn.Parameter(self.conv1x1_weights)
        # non-traceable natively spatial and temporal aggregation of remapped node features
        # split into a separate module for the quantizaiton workflow-
        # self.aggregate1 = AggregateStgcn(graph1, fifo_size, kernel_size, out_channels, stride)
        # self.aggregate2 = AggregateStgcn(graph2, fifo_size, kernel_size, out_channels, stride)
        # self.aggregate3 = AggregateStgcn(graph3, fifo_size, kernel_size, out_channels, stride)


        # normalization and dropout on main branch
        self.bn_relu = nn.Sequential(
            # LayerNorm([out_channels, 1, num_joints]),
            nn.ReLU())

        # residual branch
        if residual and not ((in_channels == out_channels) and (stride == 1)):
            print("About to do conv and LN")
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                # LayerNorm([out_channels, 1, num_joints]),
            )
        else:
            print("About to do nn.Identity")
            self.residual = nn.Identity()

        # functional quantizeable module for the addition of branches
        self.functional_add = nnq.FloatFunctional()
        self.functional_mul_zero = nnq.FloatFunctional()

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = nn.Dropout(dropout)
            print("Dropout only")
        else:
            print("Dropout and RELU")
            self.do = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout))


    def eval_(self):
        # eval is called after OfflineLayer was switched for OnlineLayer with load_state_dict()
        # self.aggregate.A *= self.edge_importance #REPLACE THIS WITH ?
        # self.aggregate1.A
        return


    def forward(self, x_res, x1, x2, x3, fifo, A1, A2, A3):
        """
        In case of buffered realtime processing, Conv2D and MMUL are done on the buffered frames,
        which mimics the kernels reuse mechanism that would be followed in hardware at the expense
        of extra memory for storing intermediate results.
        """
        tolerance = 1e-5  # Define a tolerance level

        # residual branch
        print("x2.shape is: ", x2.shape)
        res = self.residual(x_res)

        #How to sum?
        #How to handle Edge importance?? Since self.A is kept here as well?
        #Strip aggregate layer entirely? How many layers?
        #
        summed = x1 + x2 + x3
        print("shape after summation: ", summed.shape)
        # print("fifo shape before concat: ", self.fifo.shape)
        # summed = summed.sum(dim=2, keepdim=True)
        print("2nd dim summation shape: ", summed.shape)
        summed = summed.permute(0, 2, 1, 3) #N,C,L,V -> N,L,C,V
        print("shape after permute: ", summed.shape)

        # Create a selector matrix for selecting the first 8 elements out of 9 along the second dimension
        # selector = torch.zeros(9, 8)
        # for i in range(8):
        #     selector[i, i] = 1
        
        # print("Selectors are equal: ", torch.equal(selector, selector2))
        # Reshape fifo to make it suitable for matrix multiplication
        # Change shape from [1, 9, 64, 25] to [1, 64, 25, 9] to align the slicing dimension last
        fifo = fifo.reshape(1,9,64,25)
        fifo_reshaped = fifo.permute(0, 2, 3, 1).reshape(-1, 9)  # Flatten all but the last dimension for matmul

        # Perform the matrix multiplication
        selected_slices = torch.matmul(fifo_reshaped, self.selector2)
        # Reshape back to the original dimensions (with the second dimension now being 8)
        selected_slices = selected_slices.reshape(1, 64, 25, 8).permute(0, 3, 1, 2)
        # print("selected_slices.shape: ", selected_slices.shape)
        # a = fifo[:,:self.fifo_size-1]
        # if torch.allclose(a, selected_slices, atol=tolerance):
        #     print("a and selected slices are doing the same thing")
        # print()

        # fifo = torch.cat((summed, fifo[:,:self.fifo_size-1]), 1)
        print("Summed dimensions: ", summed.shape, "selected_slices dimensions: ", selected_slices.shape)
        # summed = summed.permute(0,2,3,1)
        # selected_slices = selected_slices.permute(0,2,3,1)
        print("After slicing", "Summed dimensions: ", summed.shape, "selected_slices dimensions: ", selected_slices.shape)
        fifo = torch.cat((summed.reshape(1,1,64,25), selected_slices.reshape(1,8,64,25)), dim=1)
        # fifo = fifo.permute(0, 3, 1, 2)
        # fifo = fifo.unsqueeze(dim=0)
        print("shape after fifo refresh: ", fifo.shape)
        # slice the tensor according to the temporal stride size || Code commented
        # (if stride is 1, returns the whole tensor itself)
        # fifo_reshaped = fifo.reshape(1, 9, -1)  # shape [1, 9, 64*25]
        # Perform matrix multiplication
        # sli = torch.matmul(fifo_reshaped, self.selector)
        # Reshape back to original dimensions if necessary
        # sli = sli.reshape(1, 9, 64, 25)

        # a = fifo[:,range(0, self.fifo_size, self.stride)]
        # print("Selectors are equal: ", torch.equal(fifo, a))
        a = fifo
        print("shape after fifo slice: ", a.shape)
        #####
        q = torch.sum(a, dim=(1))
        q = torch.squeeze(q, 1)

        print("q.shape: ", q.shape)
        b = self.conv1x1(a)
        # b = b.squeeze(dim=0)
        b = b.reshape(1,64,1,25)
        if torch.allclose(q, b, atol=tolerance):
            print("q and b are doing the same thing")
        ###
        print("b.shape: ", b.shape)
        x = b
        # x = b[:,:,None]

        # normalize the output of the st-gcn operation and activate
        x = self.bn_relu(x)

        # add the branches (main + residual), activate and dropout
        x = self.functional_add.add(x, res)

        return self.do(x)

        # return a1, b1, c1



