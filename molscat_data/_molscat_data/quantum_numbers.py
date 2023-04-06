from typing import NamedTuple


class Tcpld(NamedTuple):
    L: int
    F1: int
    F2: int
    F12: int
    T: int
    MT: int
    
    def J1(self):
        return self.L
    def J2(self):
        return self.F1
    def J3(self):
        return self.F2
    def J23(self):
        return self.F12
    def J123(self):
        return self.T
    def MJ123(self):
        return self.MT
    

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

class LF1F2(NamedTuple):
    L: int
    ML: int
    F1: int
    MF1: int
    F2: int
    MF2: int

    def J1(self):
        return self.L
    def MJ1(self):
        return self.ML
    def J2(self):
        return self.F1
    def MJ2(self):
        return self.MF1
    def J3(self):
        return self.F2
    def MJ3(self):
        return self.MF2
