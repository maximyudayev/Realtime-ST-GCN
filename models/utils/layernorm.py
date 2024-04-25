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
        print("Normalized shape: ", normalized_shape)


    def forward(self, x):
        
        print("self.weight.shape is: ", self.weight.shape)
        print("self.bias.shape is: ", self.bias.shape)
        print("Shape of x before mean: ", x.shape)
        # import pdb; pdb.set_trace()
        mean = torch.mean(x, dim=self.dim, keepdim=True)
        print("Dim is :", self.dim)
        var = torch.var(x, dim=self.dim, keepdim=True)
        std = torch.sqrt(var + self.eps)
        x = (x - mean) / std
        with torch.no_grad():
            x = self.weight * x + self.bias
        print("Shape of x before returning from normalization: ", x.shape)
        return x
