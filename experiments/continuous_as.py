import numpy as np
import torch
from torch.distributions.normal import Normal

observation = torch.rand(20)
print('observation', observation, observation.shape)

action_example = torch.rand(2,2)
print('action_example', action_example, action_example.shape)

action_mean_layer = torch.nn.Linear(np.prod(observation.shape), np.prod(action_example.shape))
print('action_mean_layer', action_mean_layer)

action_mean = action_mean_layer(observation)
action_mean = torch.zeros(1,4) # fake value
print('action_mean', action_mean)
action_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(action_example.shape)))
print('action_logstd', action_logstd)
action_std = torch.exp(action_logstd)
action_std = torch.ones(1,4) # fake value
print('action_std', action_std)

dist = Normal(action_mean, action_std)

action = dist.sample()
action = torch.zeros(1,4) # fake value
print('action', action)

entropy = dist.entropy()
print('entropy', entropy, entropy.sum(-1))

log_prob = dist.log_prob(action)
print('log_prob', log_prob)
prob = torch.exp(log_prob)
print('prob', prob)