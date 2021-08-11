from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]
A = imread('lion.jpg')
x = np.mean(A, -1)
U, s, VT = np.linalg.svd(x, full_matrices= False)
S = np.diag(s)
plt.figure(1)
plt.semilogy(np.diag(S))    # plot with log scaling on the y axis
plt.title('Singular Values ')
plt.xlabel("j")
plt.ylabel("log \u03C3j")
plt.show()

"""Sigma matrix
    | 5 0 0 0 0 |
    | 0 4 0 0 0 |
    | 0 0 3 0 0 |
    | 0 0 0 2 0 |
    | 0 0 0 0 1 |
    
"""