import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05, 
        elementwise_affine=True, 
        bias=True, 
        device=None,
        dtype=None):

        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, device=device, dtype=dtype)) if bias else 0
        self.eps = eps
        self.dim = torch.nonzero((torch.tensor([1, *normalized_shape]) != 1), as_tuple=True)[0].tolist()


    def forward(self, x):
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        var = torch.var(x, dim=self.dim, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x = (x - mean) / std
        x = self.weight * x + self.bias
        return x
