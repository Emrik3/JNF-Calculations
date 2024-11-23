# Linear algebra API
### Fast power calculations for matrices
The Jordan normal form (JNF) is a way to transform a matrix in a way that makes taking high powers
and using it in an exponential function if much faster.

## Documentation
### Overview
Allows easy and fast calculation matrices that are not diagonalizeable.

The Jordan normal form (JNF) is a way to transform a matrix in a way that makes taking high powers
and using it in an exponential function if much faster.

JNF is a way to make matrices "almost" diagonal, all
matrices are not diagonalizable, if this is the case JNF is used to diagonalize
it as good as possible. Good for fast calculation of high powers of a matrix,
instead of caclulating A^m (O(m\*n^3) time complexity) it only calculates 4 matrix
multiplications, the setup of finding the eigenvalues is only O(n^2+m\*n).
It is also useful for making it possible to use the matrix in an exponential function. [Further reading here](https://en.wikipedia.org/wiki/Jordan_normal_form).

The API uses the function sqrt adn factorial from the library math. Matrix, eigenvects,
eigenvals, diag, hstack, echelon_form, row_join, eye, nullspace, rank, inv and zeros from the library sympy and
assertRaises from the library unittest.

If an input is specified as matrix, it means the input should be using the
sympy.Matrix() data structure.

### Roadmap

The API of this library is frozen.

The only accepted reason to modify the API of this package
is to handle issues that can't be resolved in any other
reasonable way.

### Custom Errors
#### class NonSquareMatrixError
    class NonSquareMatrixError(Exception):
        """Custom error"""
        func __init__(message string) object
Returns error NonSquareMatrixError with input message if raised.

#### class NonSympyMatrixError
    class NonSympyMatrixError(Exception):
        """Custom error"""
        func __init__(message string) object
Returns error NonSympyMatrixError with input message if raised

#### class MatrixPowerError(Exception):
    class MatrixPowerError(Exception):
        """Custom error"""
        func __init__(message string) object
Returns error MatrixPowerError with input messige if raised

## Functions
#### func is_diagonal(A Matrix)
    func is_diagonal(A Matrix) bool
Returns True if A is diagonal, else False.

#### func diagonalize(A Matrix, calc_base bool)
    func diagonalize(A Matrix, calc_base = False) Matrix, Matrix
Returns diagonalized matrix A and, if calc_base is true,
also returns the basis matrix for that diagonalization.

#### func N_superdiag(size int)
    func N_superdiag(size int) Matrix
Returns a nilpotent matrix with ones on the superdiagonal, example of size 4 below.

[0, 1, 0, 0]

[0, 0, 1, 0]

[0, 0, 0, 1]

[0, 0, 0, 0]

#### func jnf(A Matrix, calc_base, give_blocks)
    func jnf(A Matrix, calc_base=False, give_blocks=False) Matrix, Matrix, list
Returns the Jordan Normal Form of matrix A and, if calc_base is True,
also return the basis matrix for that form. If give_blocks is True also returns
the blocks of the jnf matrix.

#### func jnf_exp(A Matrix)
    func jnf_exp(A Matrix) Matrix
Returns e to the power of the inserted matrix via jnf calculation.

#### func jnf_pow(A Matrix, pow int)
    func jnf_pow(A Matrix, pow int) Matrix
Returns the matrix A to the power of pow using jnf.
