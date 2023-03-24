import sys
from collections import namedtuple
from typing import NamedTuple
import numpy as np
from py3nj import wigner3j, clebsch_gordan

J1 = [1,2,3]
J2 = [20,30]

JJ1,JJ2 = np.meshgrid(J1,J2, indexing = 'xy')

print(JJ1)
print(JJ2)
print(JJ1.shape)

# xx = np.full((*JJ1.shape,6), [1,2,3,4,0,6], dtype = int)
xx = np.full((2,3,6), [1,2,3,4,5,6], dtype = np.int16)
print(xx)
xx = xx.transpose(2,0,1)
yy = xx.copy()
print(xx)
xx[4] = JJ1
yy[4] = JJ2
print(xx)
print(yy)

print(np.sum(JJ1), np.sum(JJ2))
print(np.sum(xx), np.sum(yy))
print(xx.dtype)
# print(xx.nbytes)

x = [0,1,2]
y = [0,1,2]
xx, yy = np.meshgrid(x,y)
print(xx)
print(yy)

f = np.vectorize(lambda x, y: 2**x * 2**y, otypes = [np.int16])
print(x,y)
fxy = f(x,y)
print(fxy, fxy.dtype)
fxy = f(xx, yy)
print(fxy)
# print(f([[2,2],[0,0]], [3,4]))
print(np.ndindex(fxy.shape))
print(list(np.ndindex(fxy.shape)))
print(np.indices(fxy.shape))

# iter = ((fxy[i,j] for j in range(fxy.shape[1])) for i range(fxy.shape[0]))
# print(iter)

# m = np.fromiter(iter, dtype = np.dtype((int,3)))
# print(m)

def xx(x):
    return 5**x

zz = np.fromiter((xx(x+2*i) for i in range(2) for x in range(5)), dtype=int).reshape(2,5)
print(zz)
zz = np.array([xx(x) for x in range(5)])
print(zz)
# fxx = f(xx)
# fyy = f(yy)
# # fxx = np.fromfunction(lambda i,j: 2**(xx[i]+j), xx.shape)
# print(fxx, fxx.dtype)
# print(fyy, fyy.dtype)


# j1,j2,j3,mj1,mj2,mj3 = 2,3,3,0,1,1
# x = (-1)**((j1-j2+mj3)/2) * np.sqrt(j3+1) *wigner3j(j1,j2,j3,mj1,mj2,-mj3)
# y = clebsch_gordan(j1,j2,j3,mj1,mj2,mj3)

# print(x,y)
# print(f"x == y: {x == y}")

# print(sys.platform)

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
class LF12(NamedTuple):
    L: int
    ML: int
    F1: int
    F2: int
    F12: int
    MF12: int

    def J1(self):
        return self.L
    def MJ1(self):
        return self.ML
    def J2(self):
        return self.F1
    def J3(self):
        return self.F2
    def J23(self):
        return self.F12
    def MJ23(self):
        return self.MF12

def test(qn_out, qn_in):
    return qn_out.J1(), qn_in.J1()

# vtest = np.vectorize(test)
# y = np.array([[2,3],[0,0]])
# x = vtest(LF12(y, 2, 3, 4, 5, 6),LF12(y, 2, 3, 4, 5, 6))
# print(x)

# iter = (())
# x = np.fromiter()