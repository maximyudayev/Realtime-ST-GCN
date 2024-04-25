import torch
import torch.nn.functional as F

a = torch.rand(1,4)
# b = torch.rand(1,64,1,25)
# c = torch.add((a,b), dim=1)

print("shape of a: ", a.shape)
print("a is: ", a)
sum = torch.sum(a, 0)
print("summed a: ", sum)
print("shape after summing: ", sum.shape)
# print("shape of b: ", b.shape)
# print("shape of c: ", c.shape)