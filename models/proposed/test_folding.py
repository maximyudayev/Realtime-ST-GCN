import torch
import torch.nn.functional as F

N = 1
C = 3
L = 47

W = 10
G = 3

P = W-(G-1)-(L-W)%(W-(G-1))

a = torch.arange(N*C*L,dtype=torch.float32).view(N,C,L)
a = F.pad(a, (0, P))
a = a.unfold(2, W, W-(G-1))

b = torch.clone(a)
b[:,:,1:,:(G-1)] = 0
b = b.permute(0,1,3,2).contiguous()
b = b.view(N,C*W,-1)
b = F.fold(b, output_size=(1,L+P), kernel_size=(1,W), stride=(1,W-(G-1)))[:,:,0,:L]
