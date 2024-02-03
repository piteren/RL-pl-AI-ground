from abc import abstractmethod

class CA:

    @property
    def num(self):
        raise NotImplementedError

ca = CA()
print(ca.num)