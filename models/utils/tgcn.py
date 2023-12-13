import torch
import torch.nn as nn

class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
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
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True):
        
        super().__init__()

        self.out_channels = out_channels
        self.partitions = partitions

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * partitions,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)


    def forward(self, x, A):
        """Broadcasted N-dimensional matrix multiplication with the 3D/4D adjacency matrix.

        Args:
            x : ``Tensor[N, C, L, V]`` 
                Input data.

            A : ``Tensor[N, P, V, V] | Tensor[P, V, V]`` 
                Adjacency matrix.
        """

        # updating feature vectors on each joint
        N,_,L,V = x.size()
        x = self.conv(x)
        x = x.view(N, self.partitions, self.out_channels*L, V)
        # (N,P*C,L,V) -> (N,P,C*L,V)

        # performing spatial accumulation of new feature vectors for each joint with A of (N,P,V,v) or (P,V,v)
        x = torch.matmul(x, A)
        # (N,P,C*L,V)
        x = torch.sum(x, dim=1).view(N, self.out_channels, L, V)
        return x
        