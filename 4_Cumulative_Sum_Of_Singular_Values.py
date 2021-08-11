from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]
A = imread('lion.jpg')
x = np.mean(A, -1)
U, s, VT = np.linalg.svd(x, full_matrices= False)
S = np.diag(s)
plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values Cumulative Sum')
plt.xlabel("j")
plt.ylabel("( 1<=j<=r \u03A3 \u03C3j ) / ( 1<=j<=m \u03A3 \u03C3j )")
plt.show()

"""Sigma matrix
    | 5 0 0 0 0 |
    | 0 4 0 0 0 |
    | 0 0 3 0 0 |
    | 0 0 0 2 0 |
    | 0 0 0 0 1 |
    
    graph is for :
    x axis : 1 , 2 , 3, 4, .... till last number of sigma
    
    y axis is :
                5 / (5 + 4 + 3 + 2 + 1)
                (5 + 4) / (5 + 4 + 3 + 2 + 1)
                (5 + 4 + 3 ) / (5 + 4 + 3 + 2 + 1)
                (5 + 4 + 3 + 2) / (5 + 4 + 3 + 2 + 1)
                . 
                . 
                . 
                so on .. till the last singular values 
                

"""