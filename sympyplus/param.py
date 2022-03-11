from sympyplus.qvector import AbstractVectorGradient, QVector
from sympyplus.fy import uflfy
import numpy as np

class Param:
    """ Defines a Param object from a list of length 2, or a Param object. The
    first item in the list is the first derivative and the second is the
    parameter without derivative. """
    def __init__(self,param):
        if isinstance(param,Param):
            self.der = param.der
            self.vec = param.vec
        else:
            if not isinstance(param,list):
                raise TypeError('Parameters must be lists.')
            elif len(param) != 2:
                raise ValueError('Parameter must be be a list of length 2.')
            self.der = param[0]
            self.vec = param[1]
    def __repr__(self) -> str:
        return f'[{repr(self.der)},{repr(self.vec)}]'
    def __eq__(self,other) -> bool:
        if not isinstance(other,Param):
            raise TypeError('Must compare Param to Param.')
        if self.der == other.der and self.vec == other.vec:
            return True
        else:
            return False
    def __add__(self,other):
        if not isinstance(other,Param):
            raise TypeError('Must be Param.')
        return Param([self.der+other.der,self.vec+other.vec])
    def __mul__(self,other):
        return Param([self.der*other,self.vec*other])
    def explode(self):
        """ Returns the Symbols of the Param as a list. Will be deprecated. """
        return [self.der[ii,jj] for ii in range(5) for jj in range(3)] + [self.vec[ii] for ii in range(5)]
    
    # Properties
    @property
    def der(self):
        return self.__der
    @property
    def vec(self):
        return self.__vec
    @property
    def symbols(self):
        """ Returns the Symbols of the Param as a NumPy array. """
        return np.array([self.der[ii,jj] for ii in range(5) for jj in range(3)] + [self.vec[ii] for ii in range(5)])
    
    # Setters
    @der.setter
    def der(self,val):
        if not isinstance(val,AbstractVectorGradient):
            raise TypeError('Must be type AbstractVectorGradient.')
        self.__der = val
    @vec.setter
    def vec(self,val):
        if not isinstance(val,QVector):
            raise TypeError('Must be type QVector')
        self.__vec = val

class GeneralForm:
    """ Returns a more general form of the Lagrangian. More parameters and their
    derivatives are allowed besides just one. The order of the GeneralForm is
    the number of pairs of parameters and their derivative. """

    def __init__(self,expr,*params,name=None):
        self.expr = expr
        self.name = name
        self.order = len(params)

        if self.order == 0:
            raise TypeError('Expression must be in terms of parameters; none entered.')

        self.params = [Param(param) for param in params]

    def __call__(self,*new_params):
        if len(new_params) == 0:
            return self.expr
        elif len(new_params) != self.order:
            raise ValueError(f'The number of params must be equal to the order of the form ({self.order}). {len(new_params)} were given.')
        else:
            return self.new_params(*new_params).expr

    def __repr__(self):
        if self.name is not None:
            return self.name
        else:
            return 'Unnamed'

    def rename(self,name):
        if not isinstance(name,str):
            raise TypeError('Name must be a string.')
        self.name = name

    def new_params(self,*new_params):
        if len(new_params) != self.order:
            raise ValueError(f'The number of new params must be equal to the order of the form ({self.order}). {len(new_params)} were given.')

        new_params = [Param(new_param) for new_param in new_params]

        # Initialize

        new_expr = self.expr

        # Function

        for kk in range(self.order):
            old_param = self.params[kk]
            new_param = new_params[kk]

            # Substitute old parameter derivative for new parameter derivative

            for jj in range(3):
                for ii in range(5):
                    old = old_param.der[ii,jj]
                    new = new_param.der[ii,jj]
                    new_expr = new_expr.subs(old,new)

            # Substitute old parameter vector for new parameter vector

            for ii in range(5):
                old = old_param.vec[ii]
                new = new_param.vec[ii]
                new_expr = new_expr.subs(old,new)

        return GeneralForm(new_expr,*new_params,name=self.name)

    @staticmethod
    def __subs_param(expr,old_param,new_param):
        """ Takes an expr, substitutese the old_param for new_param,
        then returns the resulting expr. """
        old_symbols = old_param.symbols
        new_symbols = new_param.symbols

        subs_mat = np.stack((old_symbols,new_symbols)).T

        return expr.subs(subs_mat)

    def eval(self,*params):
        """ Evaluates the GeneralForm by pluggin in the params specified,
        then returns the new expr. """
        if len(params) == 0:
            return self.expr
        if len(params) != len(self.params):
            raise ValueError('Number of args should match number of params.')
        
        expr = self.expr

        for old_param, new_param in zip(self.params, params):
            expr = self.__subs_param(expr,old_param,new_param)
        
        return expr

    def mul(self,n):
        new_expr = self.expr * n
        return GeneralForm(new_expr,*self.params,name=self.name)

    def checkParam(self,param):
        if not isinstance(param,list):
                raise TypeError('Parameters must be lists.')
        elif len(param) != 2:
            raise ValueError('Parameter must be be a list of length 2.')
        elif not isinstance(param[0],AbstractVectorGradient):
            raise TypeError('First argument of parameter must be type AbstractVectorGradient.')
        elif not isinstance(param[1],QVector):
            raise TypeError('Second argument of parameter must be type QVector.')

    def uflfy(self):
        return uflfy(self.expr)

class Lagrangian(GeneralForm):
    def __init__(self,expr,*params,name=None):
        if len(params) != 1:
            raise ValueError(f'Lagrangian must contain only one parameter. {len(params)} were given.')
        return super.__init__(self,expr,*params,name=None)