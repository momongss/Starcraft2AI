import numpy as np
from pysc2.lib.named_array import NamedNumpyArray

a = NamedNumpyArray([1, 2, 3], ["x", "y", "z"])
b = NamedNumpyArray([4, 5, 6], ["x", "y", "z"])
print(a.shape)

for i in range(len(a)):
    print(a[i])
print(a[-1])