import torch.nn as nn

class BatchNorm1d(nn.Module):
    def __init__(
        self,
        features,
        track_running_stats=False,
        device=None):

        super().__init__()
        self.norm = nn.BatchNorm1d(features, track_running_stats=track_running_stats, device=device)


    def forward(self, x):
        N, C, T, V = x.size()
        # permutes must copy the tensor over as contiguous because .view() needs a contiguous tensor
        # this incures extra overhead
        x = x.permute(0, 3, 1, 2).contiguous()
        # (N,V,C,T)
        x = x.view(N, V * C, T)
        x = self.norm(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1)
        return x
