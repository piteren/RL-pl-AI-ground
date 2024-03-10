import torch

a = torch.tensor([[2,3],[1,2],[0,0]])
print(a, a.size())
b = torch.tensor([4,5])
print(b, b.size())
c = b.expand(a.size())
print(c, c.size())