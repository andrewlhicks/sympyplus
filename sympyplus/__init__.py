""" The purpose of this module is to add additional functionality to sympy.
The functions here are intended to be used on sympy objects only. """

from typing import Type
from sympy import *
from sympyplus.operations import *
from sympyplus.fy import *
from sympyplus.qvector import *
from sympyplus.param import *
from sympyplus.variational import *
from sympyplus.form import *
from sympyplus.pde import *

class SymmetricTracelessMatrix(Matrix):
    """ Returns the symmetric traceless part of the matrix. """
    def __new__(cls,array):
        M = Matrix(array)
        M = (1/2)*(M + M.T)
        M = M - trace(M)/3*eye(3)
        return super().__new__(cls,M)

# Set up variable with respect to which we will take derivatives

x = [Symbol('x0'),Symbol('x1'),Symbol('x2')]

# END OF CODE