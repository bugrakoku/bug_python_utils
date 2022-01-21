import numpy as np
import matplotlib.pyplot as plt


def randomCLR( red_range = (10,50), green_range = (110,160), blue_range = (200,250), iSize = None):
    if iSize is None:
        return np.array((
                np.random.randint(*red_range, size = iSize),
                np.random.randint(*green_range, size = iSize),
                np.random.randint(*blue_range, size = iSize)))
    else:
        res = np.zeros((*iSize,3)) # blank image
        # shape it layer by laer
        res[:,:,0] = np.random.randint(*red_range, size= iSize)
        res[:,:,1] = np.random.randint(*green_range, size = iSize)
        res[:,:,2] = np.random.randint(*blue_range, size= iSize)
        return res

def randomStone(sSize = (50,50), sigma = 0.5):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1,1,sSize[1]), np.linspace(-1,1,sSize[0]))
    dst = np.sqrt(x*x+y*y)
    # Initializing sigma and muu
    muu = 0.000
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    return gauss > np.random.rand(*sSize)
  
def creekImage(stoneP = 0.2):
    if stoneP < 0:
        stoneP = 0.1
    elif stoneP > 1:
        stoneP = 0.99
    # init params
    stone_w =  50
    stoneNumH, stoneNumW = 10, 20

    imSize = (stone_w * stoneNumH, stone_w * stoneNumW)

    green = (50,200,50)
    red = (250,90,90)
    # empty background image for river: a blueish noisy back ground
    img = randomCLR(iSize = imSize)
    # now add stones

    gray_range = (120,140)
    # go over all blocks to add stones
    for iy in range(stoneNumH):
        for ix in range(stoneNumW):
            if np.random.rand() < stoneP: # place a stone
                rStone = randomStone((stone_w , stone_w),np.random.randint(0,3)/10.0 + .4)
                yStone, xStone = np.where(rStone == True)
                img[yStone + iy * stone_w , xStone + ix * stone_w,:] = randomCLR(gray_range,gray_range,gray_range)


    # add shores to upper and lower part of the image
    creekW = 50
    creek = np.zeros((imSize[0]+2*creekW, imSize[1], 3), dtype = np.uint8)
    creek[:creekW+imSize[0],:,:] = green
    creek[creekW+imSize[0]:,:,:] = red
    creek[creekW:creekW+imSize[0],:,:] = img
    # add a bit of noise to shoreline
    shoreMask = np.random.rand(creek.shape[0], creek.shape[1])
    shoreW = 15
    shoreP = np.linspace(0.1, 0.9, shoreW)
    for pind, pval in enumerate(shoreP): # a soft shore
        xval = np.array(np.where(shoreMask[creekW-pind,:]>pval))
        creek[creekW-pind, xval,:] = randomCLR(iSize = xval.shape)
        creek[imSize[0] + creekW + pind, xval,:] = randomCLR(iSize = xval.shape)

    return creek
