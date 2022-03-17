'''
This file contains utilities for studying images and learning
mostly focuses around numpy
    MatPrint : Prints matrices in a better way then just using print()
    ImshowMat : Shows a matrix as an image

'''

def ImportOrInstall(importStatement, importLibrary):
    '''
    tries to execute the import statement, if fails tries to install it on demand
    quits if not successful...
    '''
    try:
        exec(importStatement)
        print(f'imported {importLibrary}... moving on...')
    except:
        userChoice = input(f'{importLibrary} is not installed, would you like to install it? Y or N\n')
        if userChoice.lower() == 'y' or userChoice.lower() == 'yes':
            try: # try installing the library
                os.system(f'pip3 install {importLibrary}')
                #if installed try import one last time
                exec(importStatement)
            except:
                print(f'{importLibrary} might not be installed... please try running the app after making sure that it is installed.')
                quit()    
        else:
            print(f'please try again after installing {importLibrary}\n')
            print(f'installation hint: pip3 install {importLibrary}')

# import usual suspects
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go

def DebugPrint(mess, gottaPrint=False):
    '''
    this is seemingly stupid function but important during debugging
    the passed message is printed if gottaPrint variable is set to True
    else nothing happens, by default nothing happens... 
    how painfully useless feels this function
    '''
    if gottaPrint:
        print(mess)



def MatImshow(img,  title='Matrix as Image', showImage = True, UseMatplot = True):
    '''
    this function accepts a matrix and treats it as an image
    those matrices that are smaller in size do not look good as an image
    this function scales the images up so that they can be viewed as images
    where the pixels become larger squares
    Resulting image is shown by default, if this is not desired set showImage to False
    Resulting image is shown by Matplotlib by default, if plotly is desired, set UseMatplot to False
    '''
    # start with an empty array
    resImg = np.array([])
    # make sure that img is a proper numpy array
    if not isinstance(img, np.ndarray):
        print('ImshowMat function only accepts numpy arrays')
    elif img.ndim <2 or img.ndim > 3:
        print('ImshowMat function only accepts numpy arrays that are 2D or 3x2D, i.e. RGB')
    else: #let's go
        # let's decide on the dimensions
        # we will assume some image width where the output will be close to this value
        imgSize = 680
        imgH, imgW = img.shape
        if imgH > imgW: # imge is tall
            imgScaler = int(imgSize / imgH)
        else:
            imgScaler = int(imgSize / imgW)
        # good to go
        if img.ndim == 2: # gray scale image assumed
            # create image
            resImg = np.zeros((imgH*imgScaler, imgW*imgScaler))
            #
            # go over matrix element by element in each row, and 
            for i in range(imgH):
                for j in range(imgW):
                    imgBlock = np.ones((imgScaler, imgScaler)) * img[i,j]
                    resImg[i*imgScaler:(i+1)*imgScaler, j*imgScaler:(j+1)*imgScaler] = imgBlock
        else: # assume RGB
            print('RGB images are not implemented yet :(')
    # display if needed
    if showImage:
        if UseMatplot:
            plt.imshow(resImg, cmap=plt.get_cmap("gray"))
            plt.axis('off')
            plt.title(title)
            plt.show()
        else: # use plotly
            fig = px.imshow(resImg, color_continuous_scale='gray')
            fig.show()
    return resImg # anyway

def MatPrint(M, title = 'Matrix:'):
    '''
    this fuction accepts a variable M, expected to be in the form of np.array 
    and prints it more like a matrix
    Potential future work:
      digit size is currently fixed to 8, therefore it will not show arrays with long numbers properly
    '''
    # please do not crash
    try:
        if M.ndim != 2: # this function is meant to print only 2D arrays, if not 2D just print it
            print(title)
            print(M)
        else:
            print(title)
            spc = ' '
            sep = '|'
            for r in M: # go over each row, i.e. 1 and 2 :)
                res = sep + spc
                for d in r: # go over each element in the current row
                    res= res+ '{:8.2f}'.format(d) + spc
                res = res + sep 
                print(res)
    except:
        print(f'Is {M} really a meaningful numpy array?')

def CData(M, title='Subspace Data', viewangles = (30, 45), A= None):
    '''
    This function plots the data in M using 2D or 3D plot
    Data should be np.array
    Function does not return anything, but whines about data that cannot be plotted
    '''
    
    if M.shape[0] == 2: # then use regular plot
        # generate figure for 3D scatter
        plt.scatter(M[0,:], M[1,:], marker='*', color='red') 
        if A is not None: # there are arrows to draw
            for r in range(A.shape[0]): # go over rows
                plt.arrow(0,0,A[0,r], A[1,r], head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        plt.title(title)
        plt.show()
    elif M.shape[0] == 3: # use scatter plot
        # generate figure for 3D scatter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # add data to figure
        ax.scatter(M[0,:], M[1,:], M[2,:], c='r', marker='*')
        # add axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(title)
        if A is not None: # there are arrows to draw
            for r in range(A.shape[0]): # go over rows
                plt.arrow(0,0,A[r,0], A[r,1])
    
        # choose custom view point
        ax.view_init(*viewangles)
        # show plot... on colab or jupyter you do not need the following, but while 
        # running locally, without plt.show nothing might show up
        plt.show()
    else:
        print('CData cannot manage to make you see the data... sorry...')

def DataInSubspace(n=3,m=50, d=2, normal_dist = True):
  '''
  This function generates an nxm data matrix M, 
  where columns of M come from d-dimensional subspace
  Hence all parameters are expected to be positive where
  n >= d 
  and 
  m >= d
  Returns:
    a numpy array of shape (n,m) if you passed proper parameters
    [] if you passed improper set of parameters
  '''
  if normal_dist: # data will be drawn from a normal distribution
    genData = np.random.randn
  else: # data will be uniformly drawn from [0-1]
    genData = np.random.rand
  M = np.array([]) # let's start with an empty array
  if isinstance(n,(int,float)) and isinstance(m,(int,float)) and isinstance(d,(int,float)): # they should also be meaningful in content
    if n>0 and m>0 and d>0:
      if d <= n: # almost there
        if d <= m: # finally
          '''
          pay attention to the fact that meat of the code is here, rest is just safe guarding
          eventough all of this part can be written in a single line statement let's go step by step
          recall that our data points to live in a d-dimensional subpace 
          where data points come from n-dimensional space
          so let's generate a d-dimensional basis in n-dimensional space
          '''
          B = genData(n,d) # no more np.random.randn(n,d) or np.random.rand(n,d)
          '''
          due to the random nature of well, random, rank(B) should be d given that d<=n
          however to dodge the bullet of an ill-conditioned data matrix, let's 
          go for an orthogonal basis
          at this point if you will feel better, you can check the rank of B 
          and generate it again if rank is < d, but highly unlikely
          now that B is not square in most cases, you cannot check the determinant
          for ill-conditioned cases. Wait for SVD down the road!
          '''
          B = orth(B)
          # finally, let's generate m-many data points using B as their basis
          # now that question does not impose any norm on data, we will be happy with random
          M = np.matmul(B, genData(d,m)) # no more np.random.randn(d,m) ... 
          # note that you can also use rand function, what will change between randn and rand?
        else:
          print(f'Number of data should not be less than subspace dimension, othersise there is no way to define this subspace with that much data, check out the help:\n{DataInSubspace.__doc__}')
      else:
        print(f'Subspace dimension cannot be larger than the ambient space, check out the help:\n{DataInSubspace.__doc__}')
    else:
      print(f'You should pass positive integers to this function, check out the help:\n{DataInSubspace.__doc__}')
  else:
    print(f'You should pass numbers to this function, check out the help:\n{DataInSubspace.__doc__}')
  return M        
        
def DataFromUnionOfSubspaces(d=3,D=[1,2], N=[20,200], data_scaler=1, normal_data = True):
    '''
    This function accepts d, D, N where:
    d is the dimension of the vector space from which data is randomly drawn
    D is a list containing dimensions of subspaces
    N is also a list  and containsthe data in subspaces that correspond the 
    dimensions given in D 
    
    Optionally you can:
    - scale data using data_scaler, which is 1 by default
    - decide if you want a uniform or normal distribution 
    from which data will be chosen using normal_data, by default it uses normal distribution
    
    returns data matrix M
  
    By default function returns 20 points from a line, and 200 points from a plane in R3
    M is 3x220
  
    Another example:
    If the function is called as follows:
    DataFromUnionOfSubspaces(5, [2, 2], [200, 300])
    the function will return M such that
    M is 5x500 (200 points in the first 2D subspace, and 300 in the second)
    '''
    pass
