# Finding the singular value decomposition using numpy and scipy

# importing array function from numpy to create an object of ndarray
from numpy import array
import numpy as np

"""SciPy: 
    -solves scientific and mathematical problems
    -is an extension of NumPy
    -scipy.linalg has linear algebra abilities
    -contains svd()
"""
from scipy.linalg import svd

# defining  a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print("The matrix A is \n", A, "\n")


"""Calculating SVD matrix using svd() function
    - input: a matrix
    - output : U matrix
               Sigma diagonal matrix as a vector of singular values
               V matrix is in a transposed form
"""
U, s, VT = svd(A)
print("U matrix is :\n", U, "\n")
print("Sigma \u03A3 matrix is :\n", s, "\n")
print("Transpose of V matrix i.e. VT is :\n", VT)

