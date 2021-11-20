""" Extension of SymPy to handle expressions from PDEs. """

from typing import Type
from sympy import *
from sympyplus.operations import *
from sympyplus.fy import *
from sympyplus.qvector import *
from sympyplus.param import *
from sympyplus.variational import *
from sympyplus.form import *
from sympyplus.pde import *
from sympyplus.matrix import *

# Set up variable with respect to which we will take derivatives

x = [Symbol('x0'),Symbol('x1'),Symbol('x2')]

# END OF CODE