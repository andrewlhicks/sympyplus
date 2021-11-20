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

# FUNCTIONS

# Checks

def newtonsMethod(lhs_form,rhs_form,const_func,trial_func,test_func):
    const_func = Param(const_func)
    trial_func = Param(trial_func)
    test_func = Param(test_func)

    new_lhs_form = lhsForm(trial_func,test_func)
    for form in lhs_form.forms:
        new_lhs_form.add_form(secondVariationalDerivative(form,const_func,trial_func,test_func))

    new_rhs_form = rhsForm(test_func)
    for form in lhs_form.forms:
        new_rhs_form.add_form(form.mul(-1).new_params(const_func,test_func))
    for form in rhs_form.forms:
        new_rhs_form.add_form(form.new_params(test_func))

    return (new_lhs_form,new_rhs_form)

# CLASSES

class SymmetricTracelessMatrix(Matrix):
    """ Returns the symmetric traceless part of the matrix. """
    def __new__(cls,array):
        M = Matrix(array)
        M = (1/2)*(M + M.T)
        M = M - trace(M)/3*eye(3)
        return super().__new__(cls,M)

# Forms

class PDE_System:
    def __init__(self,domain_PDE,boundary_PDE=None):
        self.set_domain(domain_PDE)
        self.set_boundary(boundary_PDE)
    def __repr__(self):
        if self.boundary is None:
            return f'<PDE System: {self.domain.eqn_str}>'
        else:
            return f'<PDE System: {self.domain.eqn_str}\n             {self.boundary.eqn_str}>'
    @property
    def domain(self):
        return self.__domain
    @property
    def boundary(self):
        return self.__boundary


    def set_domain(self,domain_PDE):
        if domain_PDE is None:
            raise TypeError
        if not isinstance(domain_PDE,PDE):
            raise TypeError
        self.__domain = domain_PDE
    def set_boundary(self,boundary_PDE):
        if boundary_PDE is None:
            self.__boundary = None
        elif not isinstance(boundary_PDE,PDE):
            raise TypeError
        else:
            self.__boundary = boundary_PDE

class PDE(TestTrialBase):
    def __init__(self,lhs,rhs,over='domain'):
        if not isinstance(lhs,lhsForm):
            raise TypeError
        if not isinstance(rhs,rhsForm):
            raise TypeError
        if over not in ('domain','boundary'):
            raise ValueError
        if lhs.test_func != rhs.test_func:
            print(lhs.test_func,rhs.test_func)
            raise ValueError
        self.__lhs = lhs
        self.__rhs = rhs
        self.__over = over
        self.set_trial_func(lhs.trial_func)
        self.set_test_func(lhs.test_func)
    def __repr__(self) -> str:
        return f'<PDE : {self.eqn_str}>'
    @property
    def lhs(self):
        return self.__lhs
    @property
    def rhs(self):
        return self.__rhs
    @property
    def over(self):
        return self.__over
    @property
    def eqn_str(self):
        lhs = ' + '.join([repr(form) for form in self.lhs.forms])
        rhs = ' + '.join([repr(form) for form in self.rhs.forms])
        return f'[{lhs} = {rhs}] over {self.over}'

# Set up variable with respect to which we will take derivatives

x = [Symbol('x0'),Symbol('x1'),Symbol('x2')]

# END OF CODE