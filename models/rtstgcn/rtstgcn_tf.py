import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from models.utils import Graph, BatchNorm1d


class Model(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Model, self).__init__()

        self.conf = kwargs['rt-st-gcn']
        self.graph = Graph(strategy=kwargs['strategy'], **kwargs['graph'])
        self.A = tf.constant(self.graph.A, dtype=tf.float32)
        
        self.norm_in = tf.keras.layers.LayerNormalization(axis=(1, 3)) if kwargs['normalization'] == 'LayerNorm' else tf.keras.layers.BatchNormalization()
        self.fcn_in = tf.keras.layers.Conv2D(filtfiltersers=self.conf['in_ch'][0], kernel_size=(1, 1), activation='relu') #Check if this is correct
        
        self.st_gcn = []
        for i in range(self.conf['layers']):
            self.st_gcn.append(OnlineLayer(
                num_joints=kwargs['graph']['num_node'],
                in_channels=self.conf['in_ch'][i],
                out_channels=self.conf['out_ch'][i],
                kernel_size=self.conf['kernel'],
                stride=self.conf['stride'][i],
                num_partitions=self.A.shape[0],
                residual=not not self.conf['residual'][i],
                dropout=self.conf['dropout'][i],
                importance=self.conf['importance'],
                graph=self.A))

        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(1, kwargs['graph']['num_node']))
        self.fcn_out = tf.keras.layers.Conv2D(filters=kwargs['num_classes'], kernel_size=(1, 1))

    def call(self, x):
        # x = self.norm_in(x)
        x = self.fcn_in(x)
        print("shape after fcn: ", x.shape)
        for layer in self.st_gcn:
            x = layer(x)
        x = self.avg_pool(x)
        x = self.fcn_out(x)
        return tf.squeeze(x, axis=-1)
    

class OnlineLayer(tf.keras.layers.Layer):
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
        **kwargs):
        """
        """

        super(OnlineLayer, self).__init__(**kwargs)

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
        self.edge_importance = tf.Variable(tf.ones((num_partitions, num_joints, num_joints), dtype=tf.float32), trainable=False) if importance else 1

        # convolution of incoming frame
        # (out_channels is a multiple of the partition number
        # to avoid for-looping over several partitions)
        # partition-wise convolution results are basically stacked across channel-dimension
        # self.conv = nn.Conv2d(in_channels, out_channels*num_partitions, kernel_size=1, bias=False)
        self.conv = tf.keras.layers.Conv2D(out_channels*num_partitions, kernel_size=1)

        # non-traceable natively spatial and temporal aggregation of remapped node features
        # split into a separate module for the quantizaiton workflow
        self.aggregate = AggregateStgcn(graph, fifo_size, kernel_size, out_channels, stride)

        # normalization and dropout on main branch
        self.bn_relu = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(axis=[-3, -2]),
            tf.keras.layers.ReLU()
        ])

        # residual branch
        if residual and not ((in_channels == out_channels) and (stride == 1)):
            print("About to do conv and LN")
            self.residual = tf.keras.Sequential([
                tf.keras.layers.Conv2D(out_channels, kernel_size=1),
                tf.keras.layers.LayerNormalization(axis=[-3, -2])
            ])
        else:
            print("About to do nn.Identity")
            self.residual = tf.identity

        # functional quantizeable module for the addition of branches
        self.functional_add = tf.keras.layers.Add()
        self.functional_mul_zero = tf.keras.layers.Multiply()

        # activation of branch sum
        # if no resnet connection, prevent ReLU from being applied twice
        if not residual:
            self.do = tf.keras.layers.Dropout(dropout)
            print("Dropout only")
        else:
            print("Dropout and RELU")
            self.do = tf.keras.Sequential([
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(dropout)
            ])

    def call(self, x):
        """
        In case of buffered realtime processing, Conv2D and MMUL are done on the buffered frames,
        which mimics the kernels reuse mechanism that would be followed in hardware at the expense
        of extra memory for storing intermediate results.
        """

        # residual branch
        print("Shape before residual: ", x.shape)
        if not self.is_residual:
            print("I am in not self.is_residual")
            res = self.functional_mul_zero(x, 0.0)
        else:
            print("I am in else")
            res = self.residual(x)

        print("Shape after residual: ", x.shape)

        # print("in online layer. shape before conv: ", x.shape)
        # spatial convolution of incoming frame (node-wise)
        x = self.conv(x)

        # print("in online layer. shape after conv: ", x.shape)
        # aggregate features spatially and temporally (non-traceable module for FX Graph Quantization)
        x = self.aggregate(x)
        print("Aggregation done, now normalizing")
        # normalize the output of the st-gcn operation and activate
        x = self.bn_relu(x)

        # add the branches (main + residual), activate and dropout
        x = self.functional_add([x, res])

        return self.do(x)

class AggregateStgcn(tf.keras.layers.Layer):
    """
    Floating-point custom Module for spatial-temporal aggregation separated from other natively-traceable modules for the quantization workflow
    """

    def __init__(
        self,
        graph,
        fifo_size,
        kernel_size,
        out_channels,
        stride,
        **kwargs):

        super(AggregateStgcn, self).__init__(**kwargs)

        self.out_channels = out_channels
        self.stride = stride
        self.fifo_size = fifo_size
        self.kernel_size = kernel_size
        self.fifo = tf.Variable(tf.zeros((1, fifo_size, out_channels, graph.shape[1]), dtype=tf.float32), trainable=False)
        self.index = 0
        print("Graph size :", graph.shape[1])
        # FIFO for intermediate Gamma graph frames after multiplication with adjacency matrices
        # (N,G,C,V) - (N)batch, (G)amma, (C)hannels, (V)ertices
        # each layer has copy of the adjacency matrix to comply with single-input forward signature of layers for the quantization flow
        self.A = tf.constant(graph, dtype=tf.float32)

    def call(self, x):
        """
        Forward pass of the layer.
        """

        print("Shape of self.fifo: ", self.fifo.shape)

        # convert to the expected dimension order and add the partition dimension
        # reshape the tensor for multiplication with the adjacency matrix
        # (convolution output contains all partitions, stacked across the channel dimension)
        # split into separate 4D tensors, each corresponding to a separate partition
        print("shape before splitting: ", x.shape)
        print("out_channels * (x.shape[1] // out_channels): ", self.out_channels * (x.shape[1] // self.out_channels))
        x = tf.split(x, self.out_channels, axis=1)
        for tensor in x:
            print(tensor.shape)
        # concatenate these 4D tensors across the partition dimension
        x = tf.stack(x, axis=-1)
        print("Shape here: ", x.shape)
        # change the dimension order for the correct broadcasting of the adjacency matrix
        # (N,C,L,V,P) -> (N,L,P,C,V)
        x = tf.transpose(x, perm=[0, 2, 4, 1, 3])
        # single multiplication with the adjacency matrices (spatial selective addition, across partitions)
        x = tf.matmul(x, self.A)

        print("Shape after matmul: ", x.shape)

        print("shape of x.sum(axis=2): ", tf.reduce_sum(x, axis=2).shape)
        # push the frame, summed across partitions, into the FIFO
        x_sum = tf.reduce_sum(x, axis=2)
        self.fifo.assign(tf.concat((x_sum, self.fifo[:, :self.fifo_size - 1]), axis=1))

        print("shape now: ", self.fifo.shape)

        # slice the tensor according to the temporal stride size
        a = self.fifo[:, ::self.stride]

        # sum temporally
        # (C,H)
        b = tf.reduce_sum(a, axis=(1))

        # stack frame-wise tensors into the original length L
        # [(N,C,V)] -> (N,C,L,V)
        return tf.expand_dims(b, axis=-1)
