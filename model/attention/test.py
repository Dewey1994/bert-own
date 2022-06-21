import torch


a = torch.tensor((1,2,3,4,5,6,7,8))
b = torch.tensor((1,2,3,4,0,0,0,0))
a = a.masked_fill(b==0,100)
print(a)