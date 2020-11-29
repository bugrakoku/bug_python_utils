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

def MatImshow(img, showImage = True):
    '''
    this function accepts a matrix and treats it as an image
    those matrices that are smaller in size do not look good as an image
    this function scales the images up so that they can be viewed as images
    where the pixels become larger squares
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
        plt.imshow(resImg, cmap=plt.get_cmap("gray"))
        plt.show()
    return resImg # anyway

def MatPrint(M, message = 'Matrix:'):
    '''
    this fuction accepts a variable M, expected to be in the form of np.array 
    and prints it more like a matrix
    Potential future work:
      digit size is currently fixed to 8, therefore it will not show arrays with long numbers properly
    '''
    # please do not crash
    try:
        if M.ndim != 2: # this function is meant to print only 2D arrays, if not 2D just print it
            print(message)
            print(M)
        else:
            print(message)
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

def CData(M, A= None, title='Subspace Data', viewangles = (30, 45)):
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
    M = np.empty((d,0)) # start with an empty matrix or proper number of rows and 0 columns!
    # data can be randomly selected from
    try: 
        if normal_data:
            genData = np.random.randn
        else:
            genData = np.random.rand
        # let's start with loose checks - 
        # i.e. we will not check for independce and discoverability, that's up to the user
      
        if d >= max(D) and len(D) == len(N): # we are good to go
            # let's go over the element in D
            for i, Sdim in enumerate(D):  # go over subspace dimensions and add data to M
                # generate basis and normalize it
                B = genData(d,Sdim) * data_scaler
                B = normalize(B, axis=0, norm='l2')
                # generate data using B
                R = np.matmul(B, genData(Sdim, N[i]))
                # append new data as columns to M
                M = np.append(M, R, axis=1)
        # we should be done, return M
        return M
    except:
        print('crash and burn experienced in DataFromUnionOfSubspaces... go figure')
        return
