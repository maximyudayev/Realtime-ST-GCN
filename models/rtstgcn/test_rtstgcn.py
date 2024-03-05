import torch
import torch.nn.functional as F
import numpy as np

# Test dimensions
N = 1
P = 3
V = 25
C = 64
S = 1
K = 9
L = 1
G = S*(K-1)+1
D = int(V*(1+V)/2)

# FIFO for internal logic
fifo = torch.zeros(N,1,C,V,dtype=torch.int64)
#Populate the FIFO with the different numbers to see the temporal shift
for i in range(1, G):
    fr = torch.Tensor(N,1,C,V).fill_(i)
    fifo = torch.cat((fr, fifo), 1)

# Makes P V-by-V symmetric adjacency matrices
A = torch.zeros(P,V,V,dtype=torch.int64)
i,j = torch.triu_indices(V,V)
vals = torch.ones(P*D, dtype=torch.int64)*(torch.rand(P*D)>0.5)
for k in range(P):
    A[k][i,j]=vals[k*D:(k+1)*D]
    A[k].T[i,j]=vals[k*D:(k+1)*D]

# Verify adjacency matrix normalization logic
# t1 = torch.sum(A[0],0,dtype=torch.float64)
# t2 = torch.zeros(V,V,dtype=torch.float64)
# for k in range(V):
#     if t1[k]>0:
#         t2[k,k] = t1[k]**(-0.5)

# t3 = torch.tensor(A[0],dtype=torch.float64)
# t4 = torch.matmul(torch.matmul(t2, t3),t2)

# Mimic 1x1 convolution of the input

# Mimic output of spatial FC (Conv2D)
a = torch.tensor(range(N*C*P*L*V), dtype=torch.int64)
a = torch.reshape(a,(N,C*P,L,V))

# Prepare tensor for multiplication with adjacency matrices
# TODO: try replacing split->stack-permute with unfold->permute
b = torch.split(a,C,dim=1)
c = torch.stack(b,-1)
d = torch.permute(c,(0,2,4,1,3))

# Multiplication with adjacency matrices -> spatial selective addition
e = torch.matmul(d,A)

# perform temporal accumulation for each output frame
outputs = []
for i in range(e.shape[1]):
    # push the frame into the FIFO
    fifo = torch.cat((e.sum(dim=2)[:,i:i+1], fifo[:,:G-1]), 1)

    # slice the tensor according to the temporal stride size
    # (if stride is 1, returns the whole tensor itself)
    f = fifo[:,range(0, G, S)]
    print(f.shape)
    #Adaptive temporal shift
    u = 3.5 #Temporal shift parameter
    num_of_partitions = 2 * u + 1 #Number of partitions
    partition_size = int(f.shape[2] / num_of_partitions) #Size of each partition
    shifted_f = torch.reshape(f, [G*C, V]) #Initialize the shifted tensor
    u = np.ceil(u)
    shifts = [i for i in range(-int(u), int(u) + 1) if i != 0] #Create the shifts
    
    # temp_tensor = torch.eye(C, G*C)
    # temp_tensor = torch.roll(temp_tensor, -2, 0)
    # print(temp_tensor[0:G, 0:G])
    temp_tensor = torch.eye(G*C, G*C)
    for i in range(G):
        for j in range(len(shifts)):
            temp_tensor[:, i * C + (partition_size * j): i * C + (partition_size * (j + 1))] = \
            torch.roll(temp_tensor[:, i * C + (partition_size * j): i * C + (partition_size * (j + 1))], shifts[j], 0)
    
    temp_tensor[G*C - partition_size: G*C, 0:partition_size] = 0
    temp_tensor[0:partition_size, G*C - partition_size: G*C] = 0


    print(f[0][0][59] + f[0][1][3])
    shifted_f = torch.matmul(temp_tensor, shifted_f).reshape([N, G, C, V])
    print(shifted_f[0][0][63])      

    # sum temporally and across partitions
    g = torch.sum(f, dim=(1))
    print(g.shape)
    outputs.append(g)


# stack frame-wise tensors into the original length L
# [(N,C,V)] -> (N,C,L,V)
h = torch.stack(outputs, 2)

# lower triangle matrix for temporal accumulation that mimics FIFO behavior
# lt_matrix = torch.zeros(L,L,dtype=torch.int64)
# for i in range(K):
#     lt_matrix += F.pad(torch.eye(L-S*i,dtype=torch.int64), (0,i*S,i*S,0))
# lt_matrix = torch.transpose(lt_matrix,0,1)

# f_new = torch.permute(e, (0,2,3,4,1))
# g_new = torch.matmul(f_new, lt_matrix)
# # sum across partitions (N,C,V,L)
# h_new = torch.sum(g_new, dim=(1))
# # match the dimension ordering of the input (N,C,V,L) -> (N,C,L,V)
# i_new = torch.permute(h_new, (0,1,3,2))

# # compare the results of the 2 approaches (must yield identical results for integer arithmetic)
# assert(torch.equal(h, i_new))

# a.view(N,1,C,P,L,V).permute(0,1,2,4,5,3).contiguous().view(N,1,C,L,V*P)

a_temp = a.unfold(dimension=1,size=C,step=C)
A_temp = A.permute(2,0,1)[:,:,None,:,None]

temp = F.conv3d(a_temp, A_temp).permute(0,4,2,3,1)[:,:,0]

#assert(torch.equal(h, temp))