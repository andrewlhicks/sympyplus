from sympy import Matrix
from sympy import eye, trace

class SymmetricTracelessMatrix(Matrix):
    """ Returns the symmetric traceless part of the matrix. """
    def __new__(cls,array):
        M = Matrix(array)
        M = (1/2)*(M + M.T)
        M = M - trace(M)/3*eye(3)
        return super().__new__(cls,M)