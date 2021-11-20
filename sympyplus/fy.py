from sympy import Matrix, MutableDenseMatrix, ShapeError
from sympy import diag, sqrt, zeros
from sympyplus.operations import innerp

# Basis for the 3D Q-tensor

a = (sqrt(3.0)-3.0)/6.0
b = (sqrt(3.0)+3.0)/6.0
c = -sqrt(3.0)/3.0
d = sqrt(2.0)/2.0

E = [diag(a,b,c), diag(b,a,c), Matrix([[0,d,0],[d,0,0],[0,0,0]]), Matrix([[0,0,d],[0,0,0],[d,0,0]]), Matrix([[0,0,0],[0,0,d],[0,d,0]])]

# Classes

class TensorSpaceBasis(list):
    def __init__(self,basis):
        space_shape = (3,3)

        if not isinstance(basis,list):
            raise TypeError('TensorSpaceBasis must be type "list".')
        for item in basis:
            if not isinstance(item,MutableDenseMatrix):
                raise TypeError('Basis elements must be tye "Matrix".')
            if item.shape != space_shape:
                raise ShapeError(f'Basis elements must have shape {space_shape}')
        self.dim = len(basis)
        list.__init__(self,basis)

# Errors

class DimensionError(ValueError): # Changed to be subclass of ValueError rather than Exception
    pass

# Checks

def check_istensor(obj):
    """ Raises error if 'obj' is not a Sympy Matrix of shape (3,3). """
    if not isinstance(obj,MutableDenseMatrix):
        raise TypeError('Must be type MutableDenseMatrix.')
    if obj.shape != (3,3):
        raise ShapeError(f'Shape must be (3, 3), not {obj.shape}.')

def istensor(obj):
    """ Checks if 'obj' is a Sympy Matrix of shape (3,3). """
    if not isinstance(obj,MutableDenseMatrix):
        return False
    if obj.shape != (3,3):
        return False
    return True

def check_isvector(obj,dim=None):
    """ Raises error if 'obj' is not a Sympy vector of dimension 'dim'. """
    if not isinstance(obj,MutableDenseMatrix):
        raise TypeError('Must be type MutableDenseMatrix.')
    if obj.shape[1] != 1:
        raise ShapeError(f'Shape must be (*, 1), not {obj.shape}.')
    if dim is not None and obj.shape[0] != dim:
        raise ShapeError(f'Shape must be ({dim}, 1).')

def isvector(obj,dim=None):
    if not isinstance(obj,MutableDenseMatrix):
        return False
    if obj.shape[1] != 1:
        return False
    if dim is not None and obj.shape[0] != dim:
        return False
    return True

# Fys

def tensorfy(vector,basis):
    check_isvector(vector)

    if not isinstance(basis,TensorSpaceBasis):
        raise TypeError('Argument "basis" must be type TensorSpaceBasis.')
    if vector.shape[0] != basis.dim:
        raise DimensionError(f'vector and basis must have same dimension, not {vector.shape[0]} and {basis.dim}')

    tensor = zeros(3,3)

    for ii in range(basis.dim):
        tensor += vector[ii]*basis[ii]

    return tensor

def q_tensorfy(vector):
    """ Returns the Q-Tensor form of any 5D vector. """

    check_isvector(vector,5)

    q_basis = TensorSpaceBasis(E)

    return tensorfy(vector,q_basis)


def vectorfy(tensor):
    """ Returns the vector form of a Q-tensor. Checks if 'tensor' is a Q-tensor
    in the mathematical sense, then returns the corresponding vector. """

    check_istensor(tensor)

    vector = zeros(5,1)

    for ii in range(5):
        vector[ii] += innerp(tensor,E[ii])

    return vector

def uflfy(expression):
    """ Returns the UFL code for a scalar or a QVector. First checks if
    'expression' is a matrix. If not, then returns C code for the expression.
    This is a crude way to check for a scalar, but in practice it should work.
    Otherwise, checks if the expression is a 3D or 5D vector and returns the C
    code for that expression. """

    from sympy import ccode

    if not isinstance(expression,MutableDenseMatrix):
        return ccode(expression)
    elif isvector(expression,3):
        return 'as_vector([' + ','.join([ccode(expression[ii]) for ii in range(3)]) + '])'
    elif isvector(expression,5):
        return 'as_vector([' + ','.join([ccode(expression[ii]) for ii in range(5)]) + '])'
    else:
        raise TypeError('Must be a vector expression of dimension 3 or 5.')

# END OF CODE