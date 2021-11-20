from sympy import sqrt, trace

# Operations

def innerp(A,B):
    """ Returns the Frobenius inner product of two matrices. """
    return trace(A.T*B)

def outerp(A,B):
    """ Returns the "outer" or "tensor" product of two matrices. """
    return A*B.T

def mixedp(A,B):
    """ Returns the mixed product of QTensor B with the derivative of QTensor A,
    that is, epsilon(i,j,k)A(l,j,k)B(i,j) """

    product = 0

    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                for ll in range(3):
                    product += levi_civita(ii,kk,ll)*A.dx(kk)[ll,jj]*B[ii,jj]

    return product

# Norms

def fnorm(A):
    """ Returns the Frobenius norm of the matrix. """
    return sqrt(innerp(A,A))

# Other functions

def levi_civita(i,j,k):
    """ Returns the Levi-Civita symbol evaluated at indices i, j, and k, where
    the indices are in the range [0,1,2]. """

    indices = [i,j,k]

    for index in indices:
        if not (index == 0 or index == 1 or index == 2):
            raise TypeError(f'Index must be 0, 1, or 2; {index} was given.')

    def index_rearrange(indices):
        if indices[0] == indices[1] or indices[1] == indices[2] or indices[0] == indices[2]:
            return [0,0,0]
        elif indices[0] == 0:
            # print(f'Returing indices: {indices}')
            return indices
        else:
            new_indices = [0,0,0]
            for i in range(3):
                new_indices[i-1] += indices[i]
            # print(f'Old indices: {indices}, New indices: {new_indices}')
            return index_rearrange(new_indices)

    indices = index_rearrange(indices)
    # print(indices)

    if indices == [0,0,0]:
        return 0
    elif indices == [0,1,2]:
        return 1
    elif indices == [0,2,1]:
        return -1
    else:
        raise ValueError('Indices values messed up.')

# END OF CODE