import torch
from torch.distributions.categorical import Categorical


dist = Categorical(logits=torch.tensor([1,2,3,4]))
print(dist)
print(dist.probs)
print(dist.logits)
print(dist.entropy())