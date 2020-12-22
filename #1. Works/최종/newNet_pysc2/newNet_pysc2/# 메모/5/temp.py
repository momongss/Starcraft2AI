import numpy as np
from pysc2.lib.named_array import NamedNumpyArray

a = NamedNumpyArray([1,2,3], names=['a', 'b', 'c'])
print(a, type(a))

b = np.asarray(a)
print(b, type(b))