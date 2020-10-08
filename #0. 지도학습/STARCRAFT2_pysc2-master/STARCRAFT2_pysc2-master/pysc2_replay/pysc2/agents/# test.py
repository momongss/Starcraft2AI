from collections import deque
from numpy import random

a = deque()
a.append(1)
a.append(2)
a.append(3)
a.append(4)
a.append(5)
a.append(6)
a.append(7)

for _ in range(10):
    print(random.sample(a, 1))