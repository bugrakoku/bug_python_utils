'''
This file contains utilities for studying images and learning
mostly focuses around numpy
    MatPrint : Prints matrices in a better way then just using print()
    ImshowMat : Shows a matrix as an image

'''

# import usual suspects
import numpy as np
import matplotlib.pyplot as plt

def ImshowMat(img, showImage = True):
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
