import torch

#https://pythonprogramming.net/introduction-deep-learning-neural-network-pytorch/

#demo to show that torch is like numpy

x=torch.Tensor([5,3])
y=torch.Tensor([2,3])
print(x*y)

x=torch.zeros((2,3))
print(torch.shape(x))
