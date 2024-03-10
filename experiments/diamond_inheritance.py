class A:
    def __init__(self, pa):
        self.pa = pa
    def do(self):
        print(self.pa)

class B(A):
    def __init__(self, pb, **kwargs):
        super().__init__(**kwargs)
        self.pb = pb
    def do(self):
        print(self.pb)
        super().do()

class C(A):
    def __init__(self, pc, **kwargs):
        super().__init__(**kwargs)
        self.pc = pc
    def do(self):
        print(self.pc)
        super().do()

class D(B,C):
    def __init__(self, pd, **kwargs):
        super().__init__(**kwargs)
        self.pd = pd
    def do(self):
        print(self.pd)
        super().do()

class E(C,B):
    def __init__(self, pe, **kwargs):
        super().__init__(**kwargs)
        self.pe = pe
    def do(self):
        print(self.pe)
        super().do()

class F(C,B):
    def __init__(self, pe, **kwargs):
        super().__init__(**kwargs)
        self.pe = pe
    def do(self):
        print(self.pe)
        B.do(self)
        C.do(self)

"""
a = A('a')
b = B('b', pa='a')
c = C('c', pa='a')

a.do()
b.do()
c.do()
"""

d = D('d', pa='a', pb='b', pc='c')
d.do()
print()
d = E('e', pa='a', pb='b', pc='c')
d.do()
print()
d = F('f', pa='a', pb='b', pc='c')
d.do()