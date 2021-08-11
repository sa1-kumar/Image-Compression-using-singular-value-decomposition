from matplotlib.image import imread  # imread fun from image sub module
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [16, 8]    #   create a new figure will be created, and incremnet the figure number

A = imread('lion.jpg')  # load image from a file
x = np.mean(A, -1)      # mean color of a RGB image array

img = plt.imshow(x)  # creates an image from a 2-dimensional numpy array
img.set_cmap('gray')        # set the default colormap
# plt.axis('off')
plt.show()

# compression process

U, s, VT = np.linalg.svd(x, full_matrices= False)
"""The economy-size decomposition removes extra rows or columns of zeros
 from the diagonal matrix of singular values, S , 
 along with the columns in either U or V
  that multiply those zeros in the expression A = U*S*V' ."""
S = np.diag(s)


j = 0
for r in range(5,110,20):
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    # first r cols of U , first r*r block of sigma matrix , first r cols of V transposed
    plt.figure(j+1) # a new figure will be created, and the figure number will be incremented.
    j += 1
    img = plt.imshow(Xapprox)   # matrix to image 
    img.set_cmap('gray')
    # plt.axis('off')
    plt.title('At rank : ' + str(r))
    plt.show()




'''

Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r',
 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 
 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
  'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
   'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 
   'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 
   'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r'
   , 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
    'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 
    'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 
    'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
     'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot'
     , 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone',
      'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r',
       'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
        'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth',
         'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 
         'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 
         'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 
         'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r',
          'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 
          'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral',
           'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 
           'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 
           'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 
           'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 
           'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r',
            'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
             'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
'''
