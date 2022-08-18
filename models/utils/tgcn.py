# The based unit of graph convolutional networks.

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
        """TODO: verify if .view() turns multi-dimensional tensor into a matrix 
        and does an incorrect dor product.
        """

        assert A.size(1) == self.kernel_size
        # updating feature vectors on each joint
        x = self.conv(x)
        # performing spatial accumulation of new feature vectors for each joint
        n, kc, t, v = x.size()
        x = x.view(n, A.size(0), kc//A.size(0), t, v)
        # x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # !equivalent to:
        x = x.permute(0, 2, 3, 1, 4).contiguous()
        n, c, t, k, v = x.size()
        k, v, w = A.size()
        x = x.view(n * c * t, k * v)
        A = A.view(k * v, w)
        x = torch.mm(x, A)
        x = x.view(n, c, t, w)

        return x.contiguous(), A
        