import numpy as np


# Zeros Matrix
print(np.zeros((3, 3)), '\n')

# Identity Matrix
print(np.identity(3), '\n')

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
C = np.array([[2, 4, 6], [8, 10, 12], [14, 16, 18]])

# Matrix Copy
AC = A.copy()
print(AC, '\n')

# Transpose a matrix
AT = np.transpose(A)
print(AT, '\n')

# Add and Subtract
SumAC = A + C
print(SumAC, '\n')

DifCA = C - A
print(DifCA, '\n')

# Matrix Multiply
ProdAC = np.matmul(A, C)
print(ProdAC, '\n')

# Multiply a List of Matrices
arr = [A, C, A, C, A, C]
Prod = np.matmul(A, C)
num = len(arr)
for i in range(2, num):
    Prod = np.matmul(Prod, arr[i])
print(Prod, '\n')

ChkP = np.matmul(
            np.matmul(
                np.matmul(
                    np.matmul(
                        np.matmul(arr[0], arr[1]),
                        arr[2]), arr[3]), arr[4]), arr[5])
print(ChkP, '\n')

# Check Equality of Matrices
print(Prod == ChkP, '\n')
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
print(np.allclose(Prod, ChkP), '\n')

# Dot Product (follows the same rules as matrix multiplication)
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
v1 = np.array([[2, 4, 6]])
v2 = np.array([[1], [2], [3]])
ans1 = np.dot(v1, v2)
ans2 = np.dot(v1, v2)[0, 0]
print(f'ans1 = {ans1}, ans2 = {ans2}\n')

# Unitize an array
mag1 = (1*1 + 2*2 + 3*3) ** 0.5
mag2 = np.linalg.norm(v2)
norm1 = v2 / mag1
norm2 = v2 / mag2
print(f'mag1 = {mag1}, mag2 = {mag2}, they are equal: {mag1 == mag2}\n')
print(norm1, '\n')
print(norm2, '\n')
print(norm1 == norm2)
