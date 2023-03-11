import numpy as np

a = np.empty(shape=(0,2), dtype=float)
print(a)
print(a.shape)
print(a.dtype)

b = np.asarray([[0.0,0.0]])

a = np.concatenate([a,b])
print(a)
print(a.shape)
print(a.dtype)
