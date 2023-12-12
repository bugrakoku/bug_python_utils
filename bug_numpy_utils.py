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
from skimage import io
from skimage.filters import threshold_otsu as otsu
from PIL import Image, ImageDraw, ImageFont


def DebugPrint(mess, gottaPrint=False):
    '''
    this is seemingly stupid function but important during debugging
    the passed message is printed if gottaPrint variable is set to True
    else nothing happens, by default nothing happens... 
    how painfully useless feels this function
    '''
    if gottaPrint:
        print(mess)

def GenerateDataforImage(img):
    '''
    using the given (better a black and white) image name, 
    image is loaded and and binarized, 
    coordinates of the black pixels are convereted into a data matrix
    results is 2 x p where there are p pixels on the foreground
    '''
    img = io.imread(img, as_gray=True) # read as gray scale image
    img = img < otsu(img) # we have the binary image
    Yi, Xi, = np.where(img == 1)
    indexN = np.array([Xi, -Yi])*1.0
    return indexN

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

def CData(M, title='Subspace Data?', NoColor3D = False):
    '''
    This function plots the data in M using 2D or 3D plot
    Data should be np.array
    Function does not return anything, but whines about data that cannot be plotted
    Title is trivial
    NoColor3D False obviously plots all 3D data in the same color
    If NoColor3D is passed for a data matrix M that contains 270 data points on its columns
    where 100, 50 and 120 of them come from 3 different subspaces respectively,
    NoColor3D should be a list as follows:
        NoColor3D = [100, 50, 120]
    '''
    if not 'plt' in locals(): # import plt
        import matplotlib.pyplot as plt
    if not 'go' in locals(): # import go
        import plotly.graph_objects as go
    # title should be string
    if type(title) is not str:
        title = "Some plot you wanted"
        
    data2Plot = [] # start with empty list
    # done with import checks
    if M.shape[0] == 2: # then use regular plot
        # generate figure for 3D scatter
        plt.scatter(M[0,:], M[1,:], marker='*', color='red') 
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.show()
    elif M.shape[0] == 3: # use go
        if type(NoColor3D) is list or type(NoColor3D) is np.ndarray: # color plots according to the number content in list NoColor3D
            try:
                # try color printing
                # get data slice indices from NoColor3D count list
                for i,ind in enumerate(zip([0, *list(np.cumsum(NoColor3D))], np.cumsum(NoColor3D))):  
                    data2Plot.append(go.Scatter3d(x=M[0,ind[0]:ind[1]], 
                                                   y=M[1,ind[0]:ind[1]], 
                                                   z=M[2,ind[0]:ind[1]], 
                                                   name=f'Block-{i+1}', 
                                                   mode='markers', 
                                                   marker=dict(size=3)))

            except:
                # number of data is most probably not given right or something else
                # just use no color data
                data2Plot=[go.Scatter3d(x=M[0,:], y=M[1,:], z=M[2,:], name='Plane', mode='markers', marker=dict(size=3))]
        else: # no color                
            data2Plot =[go.Scatter3d(x=M[0,:], y=M[1,:], z=M[2,:], name='Plane', mode='markers', marker=dict(size=3))]
        
        # finally add title and plot

        #'''
        #fig = go.Figure(data=[go.Scatter3d(x=M[0,:], y=M[1,:], z=M[2,:], name='Plane', mode='markers', marker=dict(size=3))])
        fig = go.Figure(data = data2Plot)
        fig.update_layout(title={'text':title,
                                 'y': 0.9,
                                 'x': 0.5,
                                 'xanchor': 'center',
                                 'yanchor': 'top'})
        fig.show()
        #'''
    else:
        print('CData cannot manage to make you see the data... sorry...')
        
        
def CDataMatPlotLib(*Matrices2Plot, title='Subspace Data', viewangles = (30, 45), figSize = (10,10), A= None):
    '''
    This function plots the data in M using 2D or 3D plot
    Data should be np.array
    Function does not return anything, but whines about data that cannot be plotted
    '''
    # based on the first matrix, decide wheter it is 2 or 3D or more
    if len(Matrices2Plot) < 1:
        print("gimme something to plot")
        return
    M = Matrices2Plot[0]
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
        for M in Matrices2Plot:
            # add data to figure
            ax.scatter(M[0,:], M[1,:], M[2,:]) #, c='r', marker='*')
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
        plt.rcParams["figure.figsize"] = figSize
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

def text2mat(txt, fspace = 30, fontW = 21, fontName = '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'):
    '''
    converts the passed txt string into a data matrix, which when plotted displays the text
    Usage:
        dm, numData = text2mat(txt, fspace, fontW, fontSize, fontName)
    Input:    
        txt: string to be converted to data matrix, data in the matrix lies on the XY plane in 3D
        fspace: size of space character in pixels, defaults to 30
        fontW: width of a font window in pixels, defaults to 21
        fontName: name of the font to be used, if default does not work, provide the full path of a ttf file
    Returns:
        dm: data matrix that contains the points that spell the given text on XY plane
        numData: return the number of data points for each letter in txt
    '''
    def char2mat(c, fontname, fontsize):
        # converts character c into a 2D matrix
        BackColor = (255,255, 255)
        LPos = (1,1) # character default position
        LSize = (20,30) # size of the image that will hold the
        font = ImageFont.truetype(fontname, fontsize) # font instace created

        img = Image.new('RGB', LSize, BackColor ) # blank image
        imgPen = ImageDraw.Draw(img) # pen to draw on the blank image

        imgPen.text(LPos, c, font=font, fill=(0,0,0)) # write the character to blank image

        img.save('c.png') # save image
        imat = GenerateDataforImage('c.png') # convert image to 2D matrix
        return imat, imat.shape[1] # return 2D image and the number of data points on it
    
    fontSize = 30 # default image size
    all = [] # data points that correspond to the text in txt
    allw = [] # number of data points in each character in txt
    fhead = 0 # location of the current character in data matrix along X axis
    for t in txt: # convert all characters to data points
        c,n = char2mat(t, fontName, fontSize) # get the next character
        if n > 0: # if c is space, we will leave space but do not add data points
            if isinstance(all, list): # at first convert all to numpy array
                all = c # first instance
            else:
                c[0,:] += fhead 
                all = np.hstack((all,c)) # append the data points for the current letter
            allw.append(n) # append the number of data points for the current character
            fhead += fontW
        else:
            fhead += fspace
    
    # finally return 3D matrix with all z-coordinates being 0 along with the number of data in each cluster
    return np.vstack((all, np.zeros(all.shape[1]))), allw



def DrawOnImage(img, coord, box=False, diagonals = True, dpen = (255,0,0)):
    '''
    This funciton takes an image, and a coordinate pair in the form of ((y1,x1), (y2,x2))
    These two points can either belown to the two ends of a line, 
    or the two opposite corners of a box
    if box = False, a line is drawn between these two points,
    else a box with its diagonals are drawn. So that you do not need to worry about the center of it
    if you do not want the diagonal, make it False
    The final parameter is the pen to draw with, by default lines will be RED
    everything is drawn on a copy of the image sent, and this copy is returned
    '''
    # create a copy of the image
    img_new = img.copy()
    P1, P2 = coord # get the points out of the coordinates list
    # assume that line or box extends from P1(y1,x1) to P2(y2,x2), note that y preceeds x to be consistent with matrix indexing convention
    maxNP = max(( abs(P2[0]-P1[0]) , abs(P2[1]-P1[1])))
    if box: # then draw a box 
        img_new[P1[0], P1[1]:P2[1], :] = dpen
        img_new[P1[0]:P2[0], P2[1], :] = dpen
        img_new[P2[0], P1[1]:P2[1], :] = dpen
        img_new[P1[0]:P2[0], P1[1], :] = dpen
        if diagonals:
            img_new[ np.linspace(P1[0],P2[0], maxNP).astype('int'), np.linspace(P1[1],P2[1], maxNP).astype('int'), :] = dpen
            img_new[ np.linspace(P2[0],P1[0], maxNP).astype('int'), np.linspace(P1[1],P2[1], maxNP).astype('int'), :] = dpen
    else: # draw a line
        img_new[ np.linspace(P1[0],P2[0], maxNP).astype('int'), np.linspace(P1[1],P2[1], maxNP).astype('int'), :] = dpen
    return img_new
