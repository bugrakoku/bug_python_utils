def MatPrint(M):
  '''
  this fuction accepts a variable M, expected to be in the form of np.array 
  and prints it more like a matrix
  '''
  try:
    if M.ndim != 2: # we cannot print this properly
      print('this function is not meant to print 2D arrays')
      print('just use regular print()')
    else:
      for r in range(M.ndim): # go over each dimension, i.e. 1 or 2 :)
        spc = ' '
        sep = '|'
        res = sep + spc
        for d in M[r]: # go over each element
          res= res+ '{:8.2f}'.format(d) + spc
        res = res + sep 
        print(res)
  except:
    print(f'Is {M} really a meaningful numpy array?')
