import numpy as np

a = np.array([1,2,3])
b = np.expand_dims(a, axis=0)
print(b)