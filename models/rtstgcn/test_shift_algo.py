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