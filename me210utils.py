import numpy as np
from numpy import arccos
from numpy.linalg import norm as vector_length

import matplotlib.pyplot as plt


def MatrixVectorPlot(M,v):
    '''
    accepts a 2x2 matrix M, and a 2x1 vector v as numpy arrays
    no error check yet, use responsibly to avoid crash
    plots v and Mv to illustrate the difference
    '''
    r = M @ v # resultant vector
    
    plt.arrow(0,0, v[0], v[1], head_width=0.1, fc='g', ec='g')
    plt.arrow(0,0, r[0], r[1], head_width=0.05, fc='b', ec='b')
    plt.legend(['v', 'Mv'], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.grid()
    # find the angle between v and r
    theta = arccos(r @ v / vector_length(r) / vector_length(v)) * 180.0 / np.pi
    plt.title(f'angle between v and r = {round(theta,2)}')
    plt.rcParams["figure.figsize"] = (10,10)

def UnitSquareDeformation(M):
    '''
    accepts a 2x2 matrix M, and displays how it deforms a unit square
    no error check yet, use responsibly to avoid crash
    '''
    # Get unit square, origin is added to the end once again so that it plots a closed polygon, this it not the only way to do this but I did it this way :)
    us = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    
    res = M @ us # transform the unit square
    res3 = np.vstack([res, np.array([0,0,0,0,0])]) # in order to be able to perform cross product, add a third dimension

    # just plot the original unit square and its transformed version
    plt.plot(us[0,:], us[1,:]) # unit squres
    plt.plot(res[0,:], res[1,:]) # transformed square
    
    # draw how points on unit square moved
    for i in range(1,4): # 0,0 stays at 0,0, so only 3 points potentially move
        plt.plot([us[0,i],res[0,i]],[us[1,i],res[1,i]],linestyle = 'dashed')
    
    plt.grid()
    plt.legend(['unit square', 'transformed square','[1,0] transforms', '[1,1] transforms', '[0,1] transforms'])
    area = np.linalg.norm(np.cross(res3[:,1], res3[:,3])) # make sure you get why this is the area!
    plt.title(f'transfomed square area = {area}\n |M|={np.linalg.det(M)}')
    
