import numpy as np

a = np.zeros((1, 384)).flatten()
b = np.zeros((1, 3)).flatten()
c = np.zeros((1, 3)).flatten()

d = np.concatenate([a, b, c])
print(d.shape)