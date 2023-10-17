import numpy as np
from numpy import cos, sin, pi, round
from numpy.linalg import inv
from numpy.linalg import matrix_rank as rank


def RotMat3D(angle, around='x'):
    '''
    Returns a rotation matrix
    Default is around x, after angle send 'y', or 'z' if 'x' is not the intention
    '''
    #
    R = np.array([[1,0,0],[0, cos(angle), -sin(angle)],[0, sin(angle), cos(angle)]])
    if around == 'y':
      R = np.array([[cos(angle), 0, sin(angle)],[0,1,0], [-sin(angle), 0, cos(angle)]])
    elif around == 'z':
      R = np.array([[cos(angle), -sin(angle), 0], [sin(angle), cos(angle), 0],[0,0,1]])
    return R

def RotMat(angle):
    '''
    Returns a rotation matrix in 2D i.e. rotation is around the non-existing z-axis
    '''
    return np.array([[cos(angle), -sin(angle)],[sin(angle), cos(angle)]])

def Circle_3D(r=1, C=[0,0,0], N=100, n=0):
  '''
    takes radius r, center [x,y], number of data points N, noise level n
    and returns D which is 2xN
    by default a unit circle on XY plane with no noise is returned
    columns on D are point on the circle centered at C with radius r
    if n is not 0 then to every point in D, add np.random.rand() x n amount of noise
    return D as a numpy array
  '''
  # just to tease the in-class assignment, I will return it in a single line, which is not good python programming though!!!
  return np.array([[r*cos(theta), r*sin(theta), 0] for theta in np.linspace(-pi,pi,N)] + np.random.rand(N,3)*n + np.array(C)).T

def Square(M = [1,1], N=100):
  '''
  Generate N many random points on a square where
  Each side should have N/4 points randomly locatd on that side
  One corner of the square is at the origin, i.e. (0,0)
  where as the opposite corner from the origin is at M = [x,y]
  return D as a numpy array where D is 2xN
  '''
  # just submit this function in a file that is named as sq_studentID.py and in your implementations
  # you can only use numpy as an imported library
  # before you submit make sure the STUDENTID is replaced with yours
  # your python file should only have this function,
  # and if you have written support functions, they should be inlcuded
  # ABSOLUTELY NO TEST CODE OR ANYTHING ELSE SHOULD BE THERE
  # GET THE HABIT OF TESTING YOUR CODE BY IMPORTING IT IN A SEPERATE FILE
  try:
    M = np.array(M).reshape(2,1)
  except:
    print(f"come up with an M=[x,y], {M} did not work\n I will use default M=[1,1]")
    M = np.array([[1],[1]])
  try:
    N = int(N/4)
  except:
    print(f'N should be a number, {N} did not work, using default value of 100')
    N = 100/4

  #v1 is the vector pointing to the center of the square
  v1 = M/2
  # v2 is the 90 degrees rotated version of it
  v2 = RotMat(pi/2) @ v1
  A = np.array([[0],[0]])
  B = v1 + v2
  C = M
  D = v1 - v2

  # points on A-D
  pAD = D * np.random.rand(N)
  # point on B-C are like A-D but translated by B
  pBC = D * np.random.rand(N) + B
  # points on A-B
  pAB = B * np.random.rand(N)
  # points on C-D are like A-B but translated by D
  pCD = B * np.random.rand(N) + D

  # finally collect them all in one array
  Sq = np.hstack([pAD,pBC,pAB, pCD])
  return Sq

