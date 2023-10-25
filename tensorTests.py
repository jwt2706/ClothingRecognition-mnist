import torch
import numpy as np

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

shape = (2,3,)
rand_tensor = torch.rand(shape)

print(f"Random Tensor: \n {rand_tensor.shape} \n")
print(f"Random Tensor: \n {rand_tensor.dtype} \n")
print(f"Random Tensor: \n {rand_tensor.device} \n")

