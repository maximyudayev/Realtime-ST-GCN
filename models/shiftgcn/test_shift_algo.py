import torch
import torch.nn.functional as F

A = torch.eye(8)
shifts  = [i for i in range(-int(A.shape[0] / 2), int(A.shape[0] / 2) + 1) if i != 0]
shift = torch.nn.parameter.Parameter(torch.ones(64))
# print(shift)
for i in range(A.shape[0]):
    if shifts[i] + i < 0 or shifts[i] + i > A.shape[0] - 1:
        A[:, i] = 0
    else:
        A[:, i] = torch.roll(A[:, i], shifts[i], 0)

print(A)

# for i in range(A.shape[0]):
#     shift_amount = shifts[i]
#     if shift_amount > 0:
#         A[:-shift_amount, i] = torch.roll(A[:-shift_amount, i], shift_amount, 0)
#         A[-shift_amount:, i] = 0  # Truncate shifted out rows
#     elif shift_amount < 0:
#         A[-shift_amount:, i] = torch.roll(A[-shift_amount:, i], shift_amount, 0)
#         A[:-shift_amount, i] = 0  # Truncate shifted out rows

# print(A)
C = 1
print(A[:, C::2])
c = torch.floor(torch.tensor(torch.ones(8)))
print(c[6])
j = torch.nn.parameter.Parameter(torch.ones(3))
c = j[2]
print(type(c))
print(c.item())
print(torch.floor(torch.tensor(-1.3)))
k = 0.0

input_tensor = torch.randn(9, 64, 5, 5)

# Define the 1x1 convolutional layer
# in_channels = 64, out_channels = 64, kernel_size = (1, 1)
conv_layer = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1))

# Apply the convolutional layer to the reshaped tensor
output_tensor = conv_layer(input_tensor)

print("Output Tensor Shape:", output_tensor.shape)