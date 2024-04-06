
print('import stuff')
from bug_numpy_utils import CData
import numpy as np
print('done with imports')


# generate a matrix to plot
M = np.random.randint(10, size=(3,5000))

CData(M)

