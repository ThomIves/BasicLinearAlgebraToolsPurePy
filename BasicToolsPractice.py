import LinearAlgebraPurePython as la 


print()
la.print_matrix(la.zeros_matrix(3,3))
print()
la.print_matrix(la.identity_matrix(3))
print()
A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
la.print_matrix(A)
print()
AM = la.copy_matrix(A)
la.print_matrix(AM)
print()
AT = la.transpose(A)
la.print_matrix(AT)
print()
C = la.matrix_addition(A,AT)
la.print_matrix(C)
print()
D = la.matrix_subtraction(C, AT)
la.print_matrix(D)
print()