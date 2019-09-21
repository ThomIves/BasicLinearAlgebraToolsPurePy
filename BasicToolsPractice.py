import LinearAlgebraPurePython as la
import numpy as np


print('Shtuff from Basic Tools Post')
print()
la.print_matrix(la.zeros_matrix(3, 3))
print()
ID = la.identity_matrix(4)
la.print_matrix(ID)
print()
A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
la.print_matrix(A)
print()
AM = la.copy_matrix(A)
la.print_matrix(AM)
print()
AT = la.transpose(A)
la.print_matrix(AT)
print()
C = la.matrix_addition(A, AT)
la.print_matrix(C)
print()
D = la.matrix_subtraction(C, AT)
la.print_matrix(D)
print()
AID = la.matrix_multiply(A, ID)
la.print_matrix(AID)
print()
MatrixList = [A, AT, ID]
Prod3 = la.multiply_matrices(MatrixList)
la.print_matrix(Prod3)
print()
ATA = la.transpose(la.transpose(A))
check = la.check_matrix_equality(A, ATA)
print("A = transpose of transpose of A?", check)
print()
AdotA = la.dot_product(A, A)
print("A.A =", AdotA)
print()
Vector = [[4], [4], [4]]
unitVector = la.unitize_vector(Vector)
la.print_matrix(unitVector)
print()
print()

print('### Recursive Determinant Shtuff')
print()
A = [[-2, 2, -3],
     [-1, 1, 3],
     [2, 0, -1]]  # Matrix from wiki
Det = la.determinant_recursive(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det, 9))
print("The Numpy Determinant of A is", round(npDet, 9))
print()

A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
Det = la.determinant_recursive(A)
npDet = np.linalg.det(A)
print("Determinant of A is", Det)
print("The Numpy Determinant of A is", npDet)
print()

A = [[1, 2, 3, 4],
     [8, 5, 6, 7],
     [9, 12, 10, 11],
     [13, 14, 16, 15]]
Det = la.determinant_recursive(A)
npDet = np.linalg.det(A)
print("Determinant of A is", Det)
print("The Numpy Determinant of A is", round(npDet, 9))
print()

A = [[1, 2, 3, 4, 1],
     [8, 5, 6, 7, 2],
     [9, 12, 10, 11, 3],
     [13, 14, 16, 15, 4],
     [10, 8, 6, 4, 2]]
Det = la.determinant_recursive(A)
npDet = np.linalg.det(A)
print("Determinant of A is", Det)
print("The Numpy Determinant of A is", round(npDet, 9))
print()
print()

print('### Fast Determinant Shtuff')
print()
A = [[-2, 2, -3],
     [-1, 1, 3],
     [2, 0, -1]]  # Matrix from wiki
Det = la.determinant_fast(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det, 9))
print("The Numpy Determinant of A is", round(npDet, 9))
print()

A = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12],
     [13, 14, 15, 16]]
Det = la.determinant_fast(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det, 9)+0)
print("The Numpy Determinant of A is", round(npDet, 9))
print()

A = [[1, 2, 3, 4],
     [8, 5, 6, 7],
     [9, 12, 10, 11],
     [13, 14, 16, 15]]
Det = la.determinant_fast(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det, 9))
print("The Numpy Determinant of A is", round(npDet, 9))
print()

A = [[1, 2, 3, 4, 1],
     [8, 5, 6, 7, 2],
     [9, 12, 10, 11, 3],
     [13, 14, 16, 15, 4],
     [10, 8, 6, 4, 2]]
Det = la.determinant_fast(A)
npDet = np.linalg.det(A)
print("Determinant of A is", round(Det, 9))
print("The Numpy Determinant of A is", round(npDet, 9))
print()
