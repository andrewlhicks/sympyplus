from sympy import Symbol
from sympy import diff
from sympyplus.param import Param, GeneralForm

def variational_derivative(general_form,*params):
    if not isinstance(general_form,GeneralForm):
        raise TypeError('First positional arg must be GeneralForm')
    for param in params:
        if not isinstance(param,Param):
            raise TypeError('Second and following positional args must be Param')
    
    # compute derivative, always assume w.r.t. first param of general_form
    tau = Symbol('tau')
    expr = general_form.eval(params[0]+params[-1]*tau,*params[1:-1])
    expr = diff(expr,tau).subs(tau,0)

    return GeneralForm(expr,*params,name=f'∂({general_form.name})')

def variationalDerivative(lagrangian,*params,name=None):
    """ Given an instance of Lagrangian, returns a GeneralForm of order 2 which
    represents the first variational derivative of the Lagrangian object respect
    to the last four objects, i.e. two parameters and their derivatives. """

    if not isinstance(lagrangian,GeneralForm):
        raise TypeError('First positional argument must be type GeneralForm.')
    elif lagrangian.order != 1:
        raise ValueError('GeneralForm must be of order 1.')

    if len(params) != 2:
        raise TypeError('Must have exactly 2 parameters.')

    params = [Param(param) for param in params]

    ##########################################################################################################

    # Compute the derivative

    tau = Symbol('tau')
    expr = lagrangian([params[0].der+tau*params[1].der,params[0].vec+tau*params[1].vec])
    expr = diff(expr,tau).subs(tau,0)

    return GeneralForm(expr,[params[0].der,params[0].vec],[params[1].der,params[1].vec],name=name)

def secondVariationalDerivative(binaryform,*params,name=None):
    """ Given an instance of a GeneralForm of order 2, returns the variational
    derivative as a GeneralForm of order 3.

    Note that this in this implementation, we have for a binary form A[Q](P),
    the derivative dA[Q](P,R) with

    Param 0 = Q
    Param 1 = R
    Param 2 = P

    But a better implementation might be

    Param 0 = Q
    Param 1 = P
    Param 2 = R

    This may be fixed later.
    """

    if not isinstance(binaryform,GeneralForm):
        raise TypeError('First positional argument must be type GeneralForm.')
    elif not binaryform.order == 2:
        raise ValueError('GeneralForm must be of order 2.')

    if len(params) != 3:
        raise TypeError('Must have exactly 3 parameters.')

    params = [Param(param) for param in params]

    ##########################################################################################################

    tau = Symbol('tau')
    expr = binaryform([params[0].der+tau*params[1].der,params[0].vec+tau*params[1].vec],[params[2].der,params[2].vec])
    expr = diff(expr,tau).subs(tau,0)

    if name == None:
        name = f'Der of {binaryform.name}'

    return GeneralForm(expr,[params[0].der,params[0].vec],[params[1].der,params[1].vec],[params[2].der,params[2].vec],name=name)