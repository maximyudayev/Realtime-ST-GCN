import torch

# Test dimensions
N = 2
P = 3
V = 5
C = 6
S = 2
K = 9
L = 1
G = S*(K-1)+1
D = int(V*(1+V)/2)

# FIFO for internal logic
fifo = torch.zeros(N,G,P,C,V)

# Makes P V-by-V symmetric adjacency matrices
A = torch.zeros(P,V,V,dtype=torch.int64)
i,j = torch.triu_indices(V,V)
vals = torch.ones(P*D, dtype=torch.int64)*(torch.rand(P*D)>0.5)
for k in range(P):
    A[k][i,j]=vals[k*D:(k+1)*D]
    A[k].T[i,j]=vals[k*D:(k+1)*D]

# Verify adjacency matrix normalization logic
t1 = torch.sum(A[0],0,dtype=torch.float64)
t2 = torch.zeros(V,V,dtype=torch.float64)
for k in range(V):
    if t1[k]>0:
        t2[k,k] = t1[k]**(-0.5)

t3 = torch.tensor(A[0],dtype=torch.float64)
t4 = torch.matmul(torch.matmul(t2, t3),t2)

# Mimic 1x1 convolution of the input

# Mimic output of spatial FC (Conv2D)
a = torch.tensor(range(N*C*P*V), dtype=torch.int64)
a = torch.reshape(a,(N,C*P,L,V))

# Prepare tensor for multiplication with adjacency matrices
b = torch.split(a,C,dim=1)
c = torch.stack(b,-1)
d = torch.permute(c,(0,2,4,1,3))

# Multiplication with adjacency matrices -> spatial selective addition
e = torch.matmul(d,A)

# perform temporal accumulation for each output frame
outputs = []
for i in range(e.shape[1]):
    # push the frame into the FIFO
    fifo = torch.cat((e[:,i:i+1], fifo[:,:G-1]), 1)
    
    # slice the tensor according to the temporal stride size
    # (if stride is 1, returns the whole tensor itself)
    f = fifo[:,range(0, G, S)]

    # sum temporally and across partitions
    g = torch.sum(f, dim=(1,2))
    outputs.append(g)

# stack frame-wise tensors into the original length L
# [(N,C,V)] -> (N,C,L,V)
h = torch.stack(outputs, 2)
