import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunctionJ (X, y, theta) -> int:
    #length of 2d array X
    length = len(X)

    #Matrix multiplication: @/matmul > np.dot
    predictions = X @ theta

    #basic operations (+-*/ **) are elementwise
    sqredError = (predictions - y)**2

    #sum elements of array
    return np.sum(sqredError) / (2*length)

test1 = np.array([[1,2],[3,4]])
test2 = np.array([[2,1],[1,2]])
product = test2 @ test1
print(product)

X = np.array([[1,1], [1,2], [1,3]])
#1D array -> 2D column vector:
y = np.atleast_2d([1,2,3]).T
theta = np.atleast_2d([0,1]).T
cost = costFunctionJ(X, y, theta)
print(cost)
