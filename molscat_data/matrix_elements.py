import numpy as np

from dataclasses import dataclass

@dataclass
class AngularMomentumVector:
    c: complex
    j: float
    _m: float

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, value):
        if value not in range(-self.j, self.j+1): raise ValueError("m must be in in range \{-j, -j+1, ..., j-1, j\}")
        self._m = value

    def adjoint(self):
        attributes = vars(self).copy()
        attributes['c'] = np.conjugate(attributes['c'])
        return AngularMomentumVector(**attributes)
    
    def __mul__(self, x: complex):
        attributes = vars(self).copy()
        attributes['c'] *= x
        return AngularMomentumVector(**attributes)
    
    def __rmul__(self, x: complex):
        return self.__mul__(x)
    
    def __truediv__(self, x: complex):
        attributes = vars(self).copy()
        attributes['c'] /= x
        return AngularMomentumVector(**attributes)

    def __repr__(self):
        return f"AngularMomentumVector(c={self.c}, j={self.j}, m={self.m})"

class SpinOperator:
    @staticmethod
    def z(v: AngularMomentumVector) -> AngularMomentumVector:
        if isinstance(v, Vectors):
            return Vectors(*[SpinOperator.z(v) for v in v.vectors if SpinOperator.z(v) != 0])
        if v.m == 0:
            return 0
        return v.m * v
    
    @staticmethod
    def plus(v: AngularMomentumVector) -> AngularMomentumVector:
        if isinstance(v, Vectors):
            return Vectors(*[SpinOperator.plus(v) for v in v.vectors if SpinOperator.plus(v) != 0])
        if v.m == v.j:
            return 0
        return v.__class__(np.sqrt(v.j*(v.j+1) - v.m*(v.m+1)) * v.c, v.j, v.m+1)

    @staticmethod
    def minus(v: AngularMomentumVector) -> AngularMomentumVector:
        if isinstance(v, Vectors):
            return Vectors(*[SpinOperator.minus(v) for v in v.vectors if SpinOperator.minus(v) != 0])
        if v.m == -v.j:
            return 0
        return v.__class__(np.sqrt(v.j*(v.j+1) - v.m*(v.m-1)) * v.c, v.j, v.m-1)

class Basis:
    def __init__(self, *vectors):
        Basis.vectors = vectors
    def __repr__(self):
        return f"Basis({self.vectors})"

# @dataclass
# class Vector:
#     def __init__(self, coefficients, basis,):
#         self.basis = basis
#         self.coefficients = np.asarray(coefficients)

#     def __add__(self, other):
#         if other.basis != self.basis: return ValueError("The basis sets should be exactly the same to add to vectors.")
#         attributes = vars(self).copy()
#         attributes['coefficients'] += other.coefficients
#         return AngularMomentumVector(**attributes)

#     def __mul__(self, other):
#         if other.basis != self.basis: return ValueError("The basis sets should be exactly the same for scalar multiplying.")
#         return np.vdot(self.coefficients, other.coefficients)

#     def __repr__(self):
#         return f"Vector({vars(self)})"

class Vectors:
    def __init__(self, *vectors):
        self.vectors = vectors

    def __add__(self, other):
        if isinstance(other, AngularMomentumVector):
            return self.__class__(*self.vectors, other)
        elif isinstance(other, self.__class__):
            return self.__class__(*self.vectors, *other.vectors)
        
    def __repr__(self):
        return f"Vectors({self.vectors})"
            

def scalar_product(bra: AngularMomentumVector, ket: AngularMomentumVector) -> float:
    if type(bra) is not AngularMomentumVector or type(ket) is not AngularMomentumVector: raise TypeError("Both arguments should be AngularMomentumVector objects")
    if bra == ket:
        return float(np.conjugate(bra.c)*ket.c)
    elif bra.j == ket.j and bra.m == ket.m:
        return np.conjugate(bra.c)*ket.c
    else:
        return 0


def A_plus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1+mf/(i+0.5))*(1+B/B_hfs*(1-mf/(i+0.5)))

def A_minus_squared(mf, i = 3/2, B = 2.97, B_hfs = 2441):
    return 0.5*(1-mf/(i+0.5))*(1-B/B_hfs*(1+mf/(i+0.5)))

v1 = AngularMomentumVector(0.5+0.25j, 1, 0)
print(v1)
print(v1.adjoint())
print(5*v1)
print(v1*5)
print("_________________")
print(v1)

print(scalar_product(v1,v1))

v2 = AngularMomentumVector(0.25+0.25j, 1, 0)

print(scalar_product(v2,v1))
print(scalar_product(v1,v2))

v3 = AngularMomentumVector(0.25+0.25j, 1, 1)

print(scalar_product(v1,v3))

v4 = AngularMomentumVector(0.1, 1, 1)

# v4.m = 4
print(v4)
# print(scalar_product(v1+v2,v1))

bs = Basis(v1,v2,v3,v4)
print(bs)

vvv = Vectors(v1, v2, v3)
print(vvv)

print(SpinOperator.z(vvv))
