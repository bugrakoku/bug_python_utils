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
