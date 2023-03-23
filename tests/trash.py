import sys
from collections import namedtuple
from typing import NamedTuple
import numpy as np
from py3nj import wigner3j, clebsch_gordan

j1,j2,j3,mj1,mj2,mj3 = 2,3,3,0,1,1
x = (-1)**((j1-j2+mj3)/2) * np.sqrt(j3+1) *wigner3j(j1,j2,j3,mj1,mj2,-mj3)
y = clebsch_gordan(j1,j2,j3,mj1,mj2,mj3)

print(x,y)
print(f"x == y: {x == y}")

print(sys.platform)

# basis = ('LL', 'F1', 'FF2', 'FF12', 'TT', 'MMT')

# Qn = namedtuple('Qn', basis)
# # Qn = NamedTuple('Qn', [(name, int) for name in basis] )

# x = Qn(2,2,3,3,4,8)
# # print(x)
# print(x.F1)

# y = (2,2,3,3,4,8)

# print(x == y)

# print(type(x))

# j1 = lambda qn: qn.F1

# print(j1(x))
