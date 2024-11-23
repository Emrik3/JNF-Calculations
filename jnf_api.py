# Emrik Erikson induviduellt projekt grudat23

from sympy import *
import math
import unittest


class NonSquareMatrixError(Exception):
    """Custom Error"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NonSympyMatrixError(Exception):
    """Custom Error"""
    def __int__(self, message):
        self.message = message
        super().__init__(self.message)

class MatrixPowerError(Exception):
    """Custom Error"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class linalgebra():
    """Main API class"""
    def is_diagonal(self, A):
        if not str(type(A)) == "<class 'sympy.matrices.dense.MutableDenseMatrix'>":
            raise NonSympyMatrixError("Only the sympy.Matrix() type is allowed as input.")

        if not self.is_square(A):
            raise NonSquareMatrixError("Only square matrices have Jordan forms.")
        """Returns True if A is a diagonal matrix, else False"""

        n = math.sqrt(len(A)) # Size of matrix
        for i in range(int(n)):
            for j in range(int(n)):
                if i != j:
                    if A[i,j] != 0: # If there isn't a zero at other plae than diagonal, it is not diagonal
                        return False
        return True


    def is_square(self, A):
        if not str(type(A)) == "<class 'sympy.matrices.dense.MutableDenseMatrix'>":
            raise NonSympyMatrixError("Only the sympy.Matrix() type is allowed as input.")

        if A.shape[0] == A.shape[1]:
            return True
        else:
            return False


    def diagonalize(self, A, calc_base = False):
        """Returns diagonal matrix if it is diagonalizeable, else False"""
        if not str(type(A)) == "<class 'sympy.matrices.dense.MutableDenseMatrix'>":
            raise NonSympyMatrixError("Only the sympy.Matrix() type is allowed as input.")

        if not self.is_square(A):
            raise NonSquareMatrixError("Only square matrices have Jordan forms.")

        vec = []
        n = math.sqrt(len(A))
        eig_vecs = A.eigenvects()
        tot = 0
        create_diag = []

        for eig_vec in eig_vecs:
            for _ in range(eig_vec[1]): # range(algebraic multiplicity)
                create_diag.append(eig_vec[0])
            if eig_vec[1] == len(eig_vec[2]): # If algebraic multiplicity equals geometric multiplicity
                tot += len(eig_vec[2])

        if tot == n: # If we have n linearly independant vectors it is diagonalizable
            diag_mat = diag(*create_diag)
        else:
            diag_mat = False

        if not calc_base:
            return diag_mat


        for eig_vec in eig_vecs:
            for i in range(len(eig_vec[2])):
                vec.append(eig_vec[2][i])
        base_change_mat = self._join_vectors(vec) # Make matrix of all the eigenvectors
        return diag_mat, base_change_mat


    def _diagonalize_for_jnf(self, eig_vecs, n, calc_base = False):
        """Private for JNF because of speed (Skips eigenvector calculation since it is done in JNF.)"""
        """Returns diagonal matrix if it is diagonalizeable, else False"""
        vec = []
        tot = 0
        create_diag = []

        for eig_vec in eig_vecs:
            for _ in range(eig_vec[1]):
                create_diag.append(eig_vec[0]) # Creates list of hoe many of each eigenvalue should be put in diag

            if eig_vec[1] == len(eig_vec[2]): # If algebraic multiplicity equals geometric multiplicity
                tot += len(eig_vec[2])
        if tot == n:
            diag_mat = diag(*create_diag)
        else:
            diag_mat = False

        if calc_base and diag_mat: # If user wants to calculate base
            for eig_vec in eig_vecs:
                for i in range(len(eig_vec[2])):
                    vec.append(eig_vec[2][i])
            base_change_mat = self._join_vectors(vec)
            return diag_mat, base_change_mat
        elif calc_base:
            return diag_mat, False
        else:
            return diag_mat



    def _join_vectors(self, vecs):
        """Returns vectors joined to matrix"""
        v = vecs[0]
        for vec in vecs[1:]:
            v = v.row_join(vec)
        return v



    def _null_space_size(self, A, eig_val, alg_mult):
        """Return the size of null space of eigenspace using the null-rank therom"""
        result = [0]
        n = int(math.sqrt(len(A))) # Size of matrix
        pow = 1
        gen_eigspace = (A - eig_val*A.eye(n)).pow(pow)
        null_size = n - gen_eigspace.rank()
        pow += 1
        while result[-1] != null_size: # While not converged
            result.append(null_size)

            if null_size == alg_mult:
                break # All eigenvectors to same eigenvalue at same power.

            gen_eigspace = (A - eig_val*A.eye(n)).pow(pow) # eye is identity matrix
            null_size = n - gen_eigspace.rank() # Linalg therom null-image
            pow += 1

        return result


    def _block_calculator(self, null_sizes):
        """Calculates number of blocks of certian sizes"""
        # According to ''anmärkning 5.19 in SF1681-F5-Tex'', rank(A**0)-rank(A**1)
        # the total number of jordan blocks and rank(A**1)-rank(A**2) is the
        # total number of jordan blocks with size 2x2 or larger.
        # removing null_sizes[n-1] we get the number of block of exactly that size.
        # We use this to calculate the blocks.
        # The start is skipped because it is always equal to size of matrix.

        # Block size corresponing to A**(n-1)
        mid = [2*null_sizes[n] - null_sizes[n + 1] - null_sizes[n - 1] for n in range(1, len(null_sizes) - 1)]

        # So that no indexerror is given, if len(null_sizes) == 1 or 0 take that one value or nothing.
        end = [null_sizes[-1] - null_sizes[-2]] if len(null_sizes) > 1 else [null_sizes[0]]
        return mid + end



    def N_superdiag(self, size):
        """Creates a matrix with ones on the superdiagonal."""
        N = zeros(size) # Matrix of size size with zeros
        for i in range(size):
            for j in range(size):
                if i == j and j != size-1: # Put them at rigth position
                    N[i, j+1] = 1
        return N


    def _jordan_block(self, eig_val, size):
        """Creates a jordan block."""
        N = self.N_superdiag(size)
        block = N + eig_val * eye(size) # N + lambda * I
        return block


    def _base_helper(self, base1, base2):
        """Finds linearly independant vector in base 2 that is not in base 1"""
        if len(base1) == 0:
            return base2[0]

        for v in base2:
            _, pivots = Matrix().hstack(*(base1 + [v])).echelon_form(
                    with_pivots=True) # Pivots is the pivot columns

            if pivots[-1] == len(base1): # This means we found a linearly independant vector
                return v



    def jnf(self, A, calc_base = False):
        """Returns the jordan normal form of the matrix A and the basis for that form"""

        if not str(type(A)) == "<class 'sympy.matrices.dense.MutableDenseMatrix'>":
            raise NonSympyMatrixError("Only the sympy.Matrix() type is allowed as input.")

        if not self.is_square(A):
            raise NonSquareMatrixError("Only square matrices have Jordan forms.")


        mat_structure = []
        eig_vals = A.eigenvals() # Algebraic multiplicity is the sum of size of all blocks with eigenval lambda.
        eig_vecs = A.eigenvects() # Geomentric is the number of blocks for that eigenvalue.
        n = int(math.sqrt(len(A)))


        # If it is diagonalizeable return it
        if calc_base:
            diag, P = self._diagonalize_for_jnf(eig_vecs, n, calc_base)
            if diag:
                return diag, P
        else:
            diag = self._diagonalize_for_jnf(eig_vecs, n, calc_base)
            if diag:
                return diag

        for eig_val in sorted(eig_vals.keys()): # In acsending order
            alg_mult = eig_vals[eig_val] # Algebraic multiplicity
            null_sizes = self._null_space_size(A, eig_val, alg_mult)
            blocks = self._block_calculator(null_sizes)
            # Creates a list of the size of block and how many of that size there are. i is the size,
            # This works because of how it is indexed in _block_calculator()
            size_blocks = [(i+1, num_of_block) for i, num_of_block in enumerate(blocks)]
            size_blocks.reverse() # Puts the biggest blocks at the start of list

            # Stores the eigenvalue of the block and how big it should be.
            mat_structure.extend([(eig_val, block_size) for block_size, num_of_block in size_blocks for _ in range(num_of_block)]) # Not sure about the _ loop


        # Create jnf form using _jordan_block()
        blocks = (self._jordan_block(eig_val, size) for eig_val, size in mat_structure)
        jordan_mat = A.diag(*blocks) # Creates a diagonal matrix putting the blocks in at diagonal,


        jordan_basis = []
        if not calc_base:
            return jordan_mat


        for eig_val in sorted(eig_vals.keys()): # Sortet so that same order as eigenvalues
            eig_basis = []

            for block_eig, size in mat_structure:
                if block_eig != eig_val:
                    continue

                base1 = ((A-eig_val*eye(n))**(size-1)).nullspace() # Generalized nullspace one power lower than base 2
                base2   = ((A-eig_val*eye(n))**size).nullspace() # Generalized nullspace


                vec = self._base_helper(base1 + eig_basis, base2) # Linearly independant vector in generalized subspace 2 independant from 1
                new_vecs = [((A-eig_val*eye(n))**i)*vec for i in range(size)] # Calculates the matrix vectors from the generalized vectors from the _base_helper.

                eig_basis.extend(new_vecs)
                jordan_basis.extend(reversed(new_vecs)) # Fix order

        basis_mat = self._join_vectors(jordan_basis)
        return jordan_mat, basis_mat



    def jnf_exp(self, A):
        """Returns e to the power of matrix A"""
        n = int(math.sqrt(len(A)))
        J, P = self.jnf(A, True)
        eig_vals = []
        exp_eig_vals = []
        for i in range(n):
            for j in range(n):
                if i==j:
                    eig_vals.append(J[i,j]) # Finds eigenvalues in jnf form so they dont have to be calculated again

        D = diag(*eig_vals)
        N = J - D # Nilpotent matrix

        # Calculate exp(D)
        for eig_val in eig_vals:
            exp_eig_vals.append(exp(eig_val))
        D_exp = diag(*exp_eig_vals) # Diagonal matrix is just operation for each value on diagonal

        # Calculate exp(N)
        N_exp = eye(n)
        i = 1
        while N != zeros(n): # Nilpotent, is zero after some number of multiplications
            N_exp += N/math.factorial(i) # Taylor expansion
            N *= N
            i+=1

        return P*D_exp*N_exp*P.inv() # Using that D and N commutes we can write:
        # exp(A) = P*exp(J)*P^-1 = P*exp(D + N)*P^-1 = P*exp(D)*exp(N)*P^-1



    def jnf_pow(self, A, pow):
        """Returns A to the power of whole number pow using the jnf method"""
        if not isinstance(pow, int):
            raise MatrixPowerError("Floats as power is not supported in this function")
        if pow >= 0:
            J, P = self.jnf(A, True)
            n = int(math.sqrt(len(A)))
            N_pow = [eye(n)]
            result = zeros(n)
            pow_iter = pow

            eig_vals = []
            for i in range(n):
                for j in range(n):
                    if i==j:
                        eig_vals.append(J[i,j]) # Finds eigenvalues in jnf form so they dont have to be calculated again

            D = diag(*eig_vals)
            N = J - D # Nilpotent matrix


            # Calculate when N**i is zero
            k = 0
            while N != zeros(n) and k != pow+1: # Nilpotent, is zero after some number of multiplications
                N_pow.append(N)
                N *= N
                k += 1

            k = 0
            fac = math.factorial(pow) # Save this value for loop
            for i in N_pow:
                pow_eig_vals = []

                for eig_val in eig_vals:
                    pow_eig_vals.append(eig_val**(pow-k))
                D_pow = diag(*pow_eig_vals) # Diagonal matrix is just operation for each value on diagonal
                result += int(fac/(math.factorial(k)*math.factorial(pow-k))) * i * D_pow
                k += 1

            return P*result*P.inv()

        else: # Negative number, take the inverse of the power
            pow = abs(pow)
            J, P = self.jnf(A, True)
            n = int(math.sqrt(len(A)))
            N_pow = [eye(n)]
            result = zeros(n)
            pow_iter = pow

            eig_vals = []
            for i in range(n):
                for j in range(n):
                    if i==j:
                        eig_vals.append(J[i,j]) # Finds eigenvalues in jnf form so they dont have to be calculated again

            D = diag(*eig_vals)
            N = J - D # Nilpotent matrix


            # Calculate when N**i is zero
            k = 0
            while N != zeros(n) and k != pow+1: # Nilpotent, is zero after some number of multiplications
                N_pow.append(N)
                N *= N
                k += 1

            k = 0
            fac = math.factorial(pow) # Save this value for loop
            for i in N_pow:
                pow_eig_vals = []

                for eig_val in eig_vals:
                    pow_eig_vals.append(eig_val**(pow-k))
                D_pow = diag(*pow_eig_vals) # Diagonal matrix is just operation for each value on diagonal
                result += int(fac/(math.factorial(k)*math.factorial(pow-k))) * i * D_pow
                k += 1

            return P*result.inv()*P.inv()



class MyTestCase(unittest.TestCase):
    """Have this class to test that the right errors get raised when the input is wrong"""
    def test1(self):
        """Testcode for methods"""
        L = linalgebra()

        """Normal JNF test wihout base"""
        A = Matrix([[1, 0, -2, 0, 0],
                    [0, 1, 0, -6, 0],
                    [0, 0, 1, 0, -12],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]])

        J = L.jnf(A)

        assert J == Matrix([[1, 1, 0, 0, 0],
                            [0, 1, 1, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1]])



        """Testing jnf_exp function"""
        assert L.jnf_exp(A) == Matrix([[E, 0, -2*E, 0, 12*E],
                                       [0, E, 0, -6*E, 0],
                                       [0, 0, E, 0, -12*E],
                                       [0, 0, 0, E, 0],
                                       [0, 0, 0, 0, E]])


        """Testing jnf_pow function"""
        assert L.jnf_pow(A, 10) == Matrix([[1, 0, -20, 0, 1080],
                                           [0, 1, 0, -60, 0],
                                           [0, 0, 1, 0, -120],
                                           [0, 0, 0, 1, 0],
                                           [0, 0, 0, 0, 1]])

        assert L.jnf_pow(A, -10) == Matrix([[1, 0, 20, 0, 1320],
                                            [0, 1, 0, 60, 0],
                                            [0, 0, 1, 0, 120],
                                            [0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 1]])
        """Double checking"""
        assert L.jnf_pow(A, 10)*L.jnf_pow(A, -10) == eye(5)


        """Testing that some smaller functions return the right thing"""
        assert L.diagonalize(A) is False

        assert L.is_diagonal(A) is False

        assert L.is_square(A) is True


        """Normal JNF test with base"""
        A = Matrix([[5, 4, 2, 1],
                    [0, 1, -1, -1],
                    [-1, -1, 3, 0],
                    [1, 1, -1, 2]])

        J, P = L.jnf(A, True)


        assert J == Matrix([[1, 0, 0, 0],
                            [0, 2, 0, 0],
                            [0, 0, 4, 1],
                            [0, 0, 0, 4]])

        assert P == Matrix([[-1, 1, 1, 1],
                            [1, -1, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 1, 0]])


        """Testing jnf_exp function"""
        assert L.jnf_exp(A) == Matrix([[2*exp(4), -E + 2*exp(4), -E + exp(2) + exp(4), -E + exp(2)],
                                       [0, E, E - exp(2), E - exp(2)],
                                       [-exp(4), -exp(4), 0, 0],
                                       [exp(4), exp(4), exp(2), exp(2)]])

        """Testing jnf_pow function"""
        assert L.jnf_pow(A, 20) == Matrix([[6597069766656, 6597069766655, 5497559187455, 1048575],
                                           [0, 1, -1048575, -1048575],
                                           [-5497558138880, -5497558138880, -4398046511104, 0],
                                           [5497558138880, 5497558138880, 4398047559680, 1048576]])


        assert L.jnf_pow(A, -3) == Matrix([[1/256, -255/256, -227/256, -7/8],
                                           [0, 1, 7/8, 7/8],
                                           [3/256, 3/256, 7/256, 0],
                                           [-3/256, -3/256, 25/256, 1/8]])



        """Checking diagonal matrix without base"""
        A = Matrix([[-1, 3, -1],
                    [-3, 5, -1],
                    [-3, 3, 1]])

        D = L.jnf(A)

        assert D == Matrix([[1, 0, 0],
                            [0, 2, 0],
                            [0, 0, 2]])

        """Also test the diagonalize function even though jnf runs via it"""
        D = L.diagonalize(A)

        assert D == Matrix([[1, 0, 0],
                            [0, 2, 0],
                            [0, 0, 2]])

        """Testing pow and exp"""
        assert L.jnf_exp(A) == Matrix([[-2*exp(2) + 3*E, -3*E + 3*exp(2), E - exp(2)],
                                      [-3*exp(2) + 3*E, -3*E + 4*exp(2), E - exp(2)],
                                      [-3*exp(2) + 3*E, -3*E + 3*exp(2), E]])

        assert L.jnf_pow(A, 15) == Matrix([[-65533, 98301, -32767],
                                           [-98301, 131069, -32767],
                                           [-98301, 98301, 1]])


        """Checking diagonal matrix with base"""
        A = Matrix([[1, 2],
                    [2, 1]])
        D, P = L.jnf(A, True)

        assert D == Matrix([[-1, 0],
                            [0, 3]])

        assert P == Matrix([[-1, 1],
                           [1, 1]])

        """Also test the diagonalize function even though jnf runs via it"""
        D, P = L.diagonalize(A, True)

        assert D == Matrix([[-1, 0],
                            [0, 3]])

        assert P == Matrix([[-1, 1],
                           [1, 1]])

        """Testing nilpotent matrix creator"""
        assert L.N_superdiag(3) == Matrix([[0, 1, 0],
                                           [0, 0, 1],
                                           [0, 0, 0]])

        assert L.N_superdiag(1) == Matrix([0])

        assert L.N_superdiag(0) == Matrix([])


        """Special cases, raising to the negative one, zero or the first power."""
        assert L.jnf_pow(A, 1) == A

        assert L.jnf_pow(A, 0) == eye(2)

        assert L.jnf_pow(A, -1) == A.inv() # Inverse of matrix


        """Testing true case of is_diagonal"""
        assert L.is_diagonal(eye(4)) is True


        """Special case, non square matrix"""
        A = Matrix([[1, 3, 1], [4, 1, 3]])
        with self.assertRaises(NonSquareMatrixError) as context:
            L.jnf(A)

        assert L.is_square(A) is False


        """Special case, non matrix input"""
        A = 1
        with self.assertRaises(NonSympyMatrixError) as context:
            L.jnf(A)

        A = "String"
        with self.assertRaises(NonSympyMatrixError) as context:
            L.jnf(A)


        """Special case non integer power input"""
        A = eye(3)
        with self.assertRaises(MatrixPowerError) as context:
            L.jnf_pow(A, 0.5)


def main():
    MyTestCase().test1()

if __name__ == "__main__":
    main()
