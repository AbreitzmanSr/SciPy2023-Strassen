import sys
print(sys.version)

import numpy as np



## Strassen from https://www.geeksforgeeks.org/strassens-matrix-multiplication/
def split(matrix):
    """
    Splits a given matrix into quarters.
    Input: nxn matrix
    Output: tuple containing 4 n/2 x n/2 matrices corresponding to a, b, c, d
    """
    row, col = matrix.shape
    row2, col2 = row//2, col//2
    return matrix[:row2, :col2], matrix[:row2, col2:], matrix[row2:, :col2], matrix[row2:, col2:]
 
def strassen(x, y):
    """
    Computes matrix product by divide and conquer approach, recursively.
    Input: nxn matrices x and y
    Output: nxn matrix, product of x and y
    """
 
    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    p1 = strassen(a, f - h) 
    p2 = strassen(a + b, h)       
    p3 = strassen(c + d, e)       
    p4 = strassen(d, g - e)       
    p5 = strassen(a + d, e + h)       
    p6 = strassen(b - d, g + h) 
    p7 = strassen(a - c, e + f) 
 
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2          
    c21 = p3 + p4           
    c22 = p1 + p5 - p3 - p7 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
 
    return c


def strassenAB(x, y,crossoverCutoff):
    """
    recursion removed.  We split matrix into quadrants and then let numpy compute intermediate products and 
    re-assemble as above for matrices smaller than crossoverCutoff.
    """
 
    # Base case when size of matrices is 1x1
    if len(x) == 1:
        return x * y
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    if (len(x) >crossoverCutoff):
     p1 = strassenAB(a, f - h,crossoverCutoff) 
     p2 = strassenAB(a + b, h,crossoverCutoff)       
     p3 = strassenAB(c + d, e,crossoverCutoff)       
     p4 = strassenAB(d, g - e,crossoverCutoff)       
     p5 = strassenAB(a + d, e + h,crossoverCutoff)       
     p6 = strassenAB(b - d, g + h,crossoverCutoff) 
     p7 = strassenAB(a - c, e + f,crossoverCutoff)  
    else:
     p1 = np.matmul(a, f - h) 
     p2 = np.matmul(a + b, h)       
     p3 = np.matmul(c + d, e)       
     p4 = np.matmul(d, g - e)       
     p5 = np.matmul(a + d, e + h)       
     p6 = np.matmul(b - d, g + h) 
     p7 = np.matmul(a - c, e + f)
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2          
    c21 = p3 + p4           
    c22 = p1 + p5 - p3 - p7 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
 
    return c

def padRow(m):
  x = []
  for i in range(len(m[0])):
    x.append(0)
  return(np.vstack((m,x)))

def padRows(m,rows):
    x = []
    for i in range(len(m[0])):
        x.append(0)
    y = []
    for i in range(rows):
      y.append(x)
    return(np.vstack((m,y)))
    
def padColumn(m):
  x = []
  for i in range(len(m)):
    x.append(0)
  return(np.hstack((m,np.vstack(x))))

def padColumns(m,cols):
    x = []
    for i in range(cols):
        x.append(0)
    y = []
    for i in range(len(m)):
        y.append(x)
    return(np.hstack((m,np.vstack(y))))
                 
def deleteLastRow(m):
    return(m[:-1, :])

def deleteLastCol(m):
    return(m[:,:-1])

def divideConquer(x, y,crossoverCutoff):
    """
    standard divide and conquer which should not provide any benefit
    """
 
    # Base case when size of matrices is <= crossoverCutoff
    if len(x) <= crossoverCutoff:
        return np.matmul(x,y)
    if len(x[0])<= crossoverCutoff:
        return np.matmul(x,y)
    
    rowDim = len(x)
    colDim = len(y[0])
    if (rowDim & 1 and True or False):  #if odd row dimension then pad
        x = padRow(x)
        y = padColumn(y)
    
    if (len(x[0]) & 1 and True or False):  #if odd column dimension then pad
       x = padColumn(x)
       y = padRow(y)
        
    if (len(y[0]) & 1 and True or False):
        y = padColumn(y)
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    if (len(x) >crossoverCutoff):
     c11 = divideConquer(a, e,crossoverCutoff) + divideConquer(b, g,crossoverCutoff)
     c12 = divideConquer(a, f,crossoverCutoff) + divideConquer(b, h,crossoverCutoff)
     c21 = divideConquer(c, e,crossoverCutoff) + divideConquer(d, g,crossoverCutoff)
     c22 = divideConquer(c, f,crossoverCutoff) + divideConquer(d, h,crossoverCutoff)
    else:
     c11 = np.matmul(a,e) + np.matmul(b,g)
     c12 = np.matmul(a,f) + np.matmul(b,h)       
     c21 = np.matmul(c,e) + np.matmul(d,g)      
     c22 = np.matmul(c,f) + np.matmul(d,h)    
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    
    x = len(c) - rowDim
    if (x > 0):
        c = c[:-x, :]  #delete padded rows

    x = len(c[0]) - colDim
    if (x > 0):
        c = c[:,:-x]  #delete padded columns
    
    return c
def strassenGeneral(x, y,crossoverCutoff):
    """
    complete recursion removed.  We split matrix into quadrants and then let numpy compute intermediate products and 
    re-assemble as above for matrices smaller than crossoverCutoff.
    """
 
    # Base case when size of matrices is <= crossoverCutoff
    if len(x) <= crossoverCutoff:
        return np.matmul(x,y)
    if len(x[0])<= crossoverCutoff:
        return np.matmul(x,y)
    
    rowDim = len(x)
    colDim = len(y[0])
    if (rowDim & 1 and True or False):  #if odd row dimension then pad
        x = padRow(x)
        y = padColumn(y)
    
    if (len(x[0]) & 1 and True or False):  #if odd column dimension then pad
       x = padColumn(x)
       y = padRow(y)
        
    if (len(y[0]) & 1 and True or False):
        y = padColumn(y)
 
    # Splitting the matrices into quadrants. This will be done recursively
    # until the base case is reached.
    a, b, c, d = split(x)
    e, f, g, h = split(y)
 
    # Computing the 7 products, recursively (p1, p2...p7)
    if (len(x) >crossoverCutoff):
     p1 = strassenGeneral(a, f - h,crossoverCutoff) 
     p2 = strassenGeneral(a + b, h,crossoverCutoff)       
     p3 = strassenGeneral(c + d, e,crossoverCutoff)       
     p4 = strassenGeneral(d, g - e,crossoverCutoff)       
     p5 = strassenGeneral(a + d, e + h,crossoverCutoff)       
     p6 = strassenGeneral(b - d, g + h,crossoverCutoff) 
     p7 = strassenGeneral(a - c, e + f,crossoverCutoff)  
    else:
     p1 = np.matmul(a, f - h) 
     p2 = np.matmul(a + b, h)       
     p3 = np.matmul(c + d, e)       
     p4 = np.matmul(d, g - e)       
     p5 = np.matmul(a + d, e + h)       
     p6 = np.matmul(b - d, g + h) 
     p7 = np.matmul(a - c, e + f)
 
    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6 
    c12 = p1 + p2          
    c21 = p3 + p4           
    c22 = p1 + p5 - p3 - p7 
 
    # Combining the 4 quadrants into a single matrix by stacking horizontally and vertically.
    c = np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))
    
    x = len(c) - rowDim
    if (x > 0):
        c = c[:-x, :]  #delete padded rows

    x = len(c[0]) - colDim
    if (x > 0):
        c = c[:,:-x]  #delete padded columns
    
    return c


import random
#create an nxn matrix of random integers between 0 and 99
def randomSquareMatrix(n):
  m = []
  for i in range(n):
    x = []
    for j in range(n):
        x.append(random.randint(0, 100))
    m.append(x)
  return(np.array(m))

#create an mxn matrix of random ints 
def randomMxNmatrix(m,n,maxx):
  mm = []
  for i in range(m):
    x = []
    for j in range(n):
        t = random.randint(0,maxx)
        x.append(t)
    mm.append(x)
  return(np.array(mm))

##create an nxn 0 matrix
def zeroMatrix(n):
  m = []
  for i in range(n):
    x = []
    for j in range(n):
        x.append(0)
    m.append(x)
  return(np.array(m))    


##traditional matrix multiplication
def multiply(a,b):
  n = len(a)
  c = zeroMatrix(n)
  for i in range(n):
    for j in range(n):
        for k in range(n):
            c[i][j] += a[i][k]*b[k][j];
  return(c)


#check for strassen accuracy
print("\ntesting accuracy of each method. if any false below we have trouble.\n")
for i in range(3):
  c = random.randint(1000, 2000)
  a = randomMxNmatrix(random.randint(1000, 2000),c,100000)
  b = randomMxNmatrix(c,random.randint(1000, 2000),100000)
  ans1 = np.matmul(a,b)  
  ans2 = strassenGeneral(a,b,128)
  print(i,len(a),len(a[0]),len(b),len(b[0]),np.allclose(ans1,ans2))
  ans3 = np.dot(a,b)
  print(np.array_equal(ans2,ans3))
  ans4 = a @ b
  ans5 = divideConquer(a,b,64)
  print(np.array_equal(ans4,ans5))



#check for strassen timings
import time
for i in range(2):
  c = random.randint(1000, 2000)
  a = randomMxNmatrix(random.randint(1000, 2000),c,100000)
  b = randomMxNmatrix(c,random.randint(1000, 2000),100000)
 
  print(i,len(a),len(a[0]),len(b[0]))

  start = time.perf_counter() 
  ans1 = np.matmul(a,b) 
  stop = time.perf_counter()
  print("numpy (seconds) ",stop - start)

  start = time.perf_counter() 
  ans1 = np.dot(a,b) 
  stop = time.perf_counter()
  print("numpyDot (seconds) ",stop - start)

  start = time.perf_counter() 
  ans1 = a @ b
  stop = time.perf_counter()
  print("a @ b (seconds) ",stop - start)

  start = time.perf_counter() 
  ans2 = strassenGeneral(a,b,64) 
  stop = time.perf_counter()
  print("strassen64 (seconds) ",stop - start)

  start = time.perf_counter() 
  ans2 = strassenGeneral(a,b,128) 
  stop = time.perf_counter()
  print("strassen128 (seconds) ",stop - start)
  
  start = time.perf_counter() 
  ans2 = strassenGeneral(a,b,256) 
  stop = time.perf_counter()
  print("strassen256 (seconds) ",stop - start)
    
  start = time.perf_counter() 
  ans2 = divideConquer(a,b,64) 
  stop = time.perf_counter()
  print("DC64 (seconds) ",stop - start)

  start = time.perf_counter() 
  ans2 = divideConquer(a,b,128) 
  stop = time.perf_counter()
  print("DC128 (seconds) ",stop - start)
  
  start = time.perf_counter() 
  ans2 = divideConquer(a,b,256) 
  stop = time.perf_counter()
  print("DC256 (seconds) ",stop - start)    



## Let's do some timings
import time

k=1
n = 128
while(k < 9):
  print("\nmatrix size "+str(n))

  a = randomSquareMatrix(n)
  b = randomSquareMatrix(n)

  start = time.perf_counter() 
  c = np.matmul(a,b)
  stop = time.perf_counter()
  print("numpy time (seconds) ",stop - start)

  start = time.perf_counter() 
  c = strassen(a,b)
  stop = time.perf_counter()
  print("strassen time (seconds) ",stop - start)

  start = time.perf_counter() 
  c = strassenAB(a,b,16)
  stop = time.perf_counter()
  print("strassen16 time (seconds) ", stop - start)
      
  start = time.perf_counter() 
  c = strassenAB(a,b,32)
  stop = time.perf_counter()
  print("strassen32 time (seconds) ", stop - start)

  start = time.perf_counter() 
  c = strassenAB(a,b,64)
  stop = time.perf_counter()
  print("strassen64 time (seconds) ", stop - start)
      
  start = time.perf_counter() 
  c = strassenAB(a,b,128)
  stop = time.perf_counter()
  print("strassen128 time (seconds) ", stop - start)

  start = time.perf_counter() 
  c = strassenAB(a,b,256)
  stop = time.perf_counter()
  print("strassen256 time (seconds) ", stop - start)
  
  start = time.perf_counter() 
  c = strassenAB(a,b,512)
  stop = time.perf_counter()
  print("strassen512 time (seconds) ", stop - start)

  start = time.perf_counter() 
  c = multiply(a,b)
  stop = time.perf_counter()
  print("naive time (seconds) ",stop - start)

  n = 2*n
  k+=1