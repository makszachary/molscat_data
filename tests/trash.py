import sys
from collections import namedtuple
from typing import NamedTuple
from pyshtools import Wigner3j

print(Wigner3j(1,2,1,1,-2))

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
