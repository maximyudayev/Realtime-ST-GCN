import torch

# Test dimensions
P = 3
V = 5
C = 6
G = 9
D = int(V*(1+V)/2)

# FIFO for internal logic
fifo = torch.zeros(G,P,C,V)

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

# Mimic output of spatial FC
a = torch.tensor(range(1*C*P*V*1), dtype=torch.int64)
a = torch.reshape(a,(1,C*P,V,1))

# Prepare tensor for multiplication with adjacency matrices
b = torch.split(a,C,dim=1)
c = torch.cat(b,0)
d = torch.permute(c,(3,0,1,2))

# Multiplication with adjacency matrices -> spatial selective addition
e = torch.matmul(d,A)

# Updating FIFO contents
f = torch.cat((e, fifo[:G-1]), 0)

# Temporal reduction and matching to input dimension
g = torch.sum(f,(0,1))
h = torch.permute(g,(1,0))[None,:]
