""" The purpose of this module is to add additional functionality to sympy.
The functions here are intended to be used on sympy objects only. """

from typing import Type
from sympy import *
from sympyplus.operations import *
from sympyplus.fy import *
from sympyplus.qvector import *
from sympyplus.param import *
from sympyplus.variational import *

# FUNCTIONS

# Checks

def is_linear_param(expression,parameter):
    """ Checks if the expression is linear in this single parameter. """
    if not isinstance(parameter,Param):
        raise TypeError('Parameter given must be type Param.')
    args = parameter.explode()

    # Begin with the second derivative test for all parameters

    for arg in args:
        second_derivative = diff(expression,arg,2)
        if not second_derivative.is_zero:
            return False

    # Next, set all args equal to zero and see if expression becomes zero

    for arg in args:
        expression = expression.subs(arg,0)
    if not expression.is_zero:
        return False

    # If the two tests do not return False, return True

    return True

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

class EnergyForm:
    def __init__(self,*params,domain=None,boundary=None):
        # Type checking

        if len(params) != 3:
            raise TypeError('EnergyForm requires exactly 3 Params.')
        if domain is None:
            domain = []
        if boundary is None:
            boundary = []
        if not isinstance(domain,list) or not isinstance(boundary,list):
            raise TypeError
        for item in domain + boundary:
            if not isinstance(item,GeneralForm):
                raise TypeError
        
        # Initialize params

        self.__params = [Param(param) for param in params]

        # Initialize domain
        self.__domain_0, self.__domain_1, self.__domain_2 = [], [], []
        self.add_domain(*domain)

        # Initialize boundary
        self.__boundary_0, self.__boundary_1, self.__boundary_2 = [], [], []
        self.add_boundary(*boundary)

    def __repr__(self):
        return f'<EnergyForm d={self.domain} b={self.boundary}>'
    
    @property
    def domain(self):
        return [repr(form) for form in self.__domain_0]
    @property
    def boundary(self):
        return [repr(form) for form in self.__boundary_0]

    @property
    def params(self):
        return self.__params

    @property
    def domain_0(self):
        return [form.uflfy() for form in self.__domain_0]
    @property
    def domain_1(self):
        return [form.uflfy() for form in self.__domain_1]
    @property
    def domain_2(self):
        return [form.uflfy() for form in self.__domain_2]
    @property
    def boundary_0(self):
        return [form.uflfy() for form in self.__boundary_0]
    @property
    def boundary_1(self):
        return [form.uflfy() for form in self.__boundary_1]
    @property
    def boundary_2(self):
        return [form.uflfy() for form in self.__boundary_2]

    @property
    def ufl_dict(self):
        return [
            {'domain':self.domain_0,'boundary':self.boundary_0},
            {'domain':self.domain_1,'boundary':self.boundary_1},
            {'domain':self.domain_2,'boundary':self.boundary_2}
        ]
    
    def differentiate_forms(self,*forms):
        # Preliminary checks
        for form in forms:
            if not isinstance(form,GeneralForm):
                raise TypeError('"forms" must be a list of items of type GeneralForm.')
            if form.order != 1:
                raise ValueError
        # Get derivatives of forms
        forms_0 = list(forms)
        forms_1 = [variationalDerivative(form,self.params[0],self.params[1]) for form in forms_0]
        forms_2 = [secondVariationalDerivative(form,self.params[0],self.params[2],self.params[1]) for form in forms_1]
        # Return derivatives
        return (forms_0, forms_1, forms_2)
    def add_domain(self,*forms):
        # Exit function if no forms given
        if forms == ():
            return
        # Differentiate forms
        forms_0, forms_1, forms_2 = self.differentiate_forms(*forms)
        # Add to existing forms
        self.__domain_0.extend(forms_0)
        self.__domain_1.extend(forms_1)
        self.__domain_2.extend(forms_2)
    def add_boundary(self,*forms):
        # Exit function if no forms given
        if forms == ():
            return
        # Differentiate forms
        forms_0, forms_1, forms_2 = self.differentiate_forms(*forms)
        # Add to existing forms
        self.__boundary_0.extend(forms_0)
        self.__boundary_1.extend(forms_1)
        self.__boundary_2.extend(forms_2)

class TestTrialBase: # Not intended for use except as base class
    def set_test_func(self,test_func):
        if test_func is None:
            raise TypeError
        self.__test_func = Param(test_func)
    def set_trial_func(self,trial_func=None):
        if trial_func is None:
            self.__trial_func = None
        else:
            self.__trial_func = Param(trial_func)
    @property
    def test_func(self):
        return self.__test_func
    @property
    def trial_func(self):
        return self.__trial_func

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

class lhsForm(TestTrialBase):
    def __init__(self,trial_func,test_func,name=None,forms=None):
        if forms is None:
            forms = []
        if not isinstance(forms,list):
            raise TypeError('\'forms\' must be a List of items of type GeneralForm.')
        self.set_trial_func(trial_func)
        self.set_test_func(test_func)
        self.name = name
        self.forms = []
        self.add_form(*forms)

    def __call__(self):
        expr = 0
        for form in self.forms:
            if not is_linear_param(form.expr,self.trial_func):
                raise ValueError(f'The form \'{form}\' of \'{self}\' is nonlinear in the trial function.')
            elif not is_linear_param(form.expr,self.test_func):
                raise ValueError(f'The form \'{form}\' of \'{self}\' is nonlinear in the test function.')
            expr += form.expr
        return uflfy(expr)

    def __repr__(self):
        return 'Untitled lhsForm' if self.name == None else f'lhsForm {self.name}'

    def add_form(self,*forms):
        for form in forms:
            if not isinstance(form,GeneralForm):
                raise TypeError('Form must be type GeneralForm.')
            self.forms.append(form)

class rhsForm(TestTrialBase):
    def __init__(self,test_func,name=None,forms=None):
        if forms is None:
            forms = []
        if not isinstance(forms,list):
            raise TypeError('\'forms\' must be a List of items of type GeneralForm.')
        self.set_test_func(test_func)
        self.set_trial_func()
        self.name = name
        self.forms = []
        self.add_form(*forms)

    def __call__(self):
        expr = 0
        for form in self.forms:
            if not is_linear_param(form.expr,self.test_func):
                raise ValueError(f'The form \'{form}\' of \'{self}\' is nonlinear.')
            expr += form.expr
        return uflfy(expr)

    def __repr__(self):
        return 'Untitled rhsForm' if self.name == None else f'rhsForm {self.name}'

    def add_form(self,*forms):
        for form in forms:
            if not isinstance(form,GeneralForm):
                raise TypeError('Form must be type GeneralForm.')
            self.forms.append(form)

# Set up variable with respect to which we will take derivatives

x = [Symbol('x0'),Symbol('x1'),Symbol('x2')]

# END OF CODE