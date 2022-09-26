import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import sklearn

def costJ(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    m = len(y)
    error = (X @ theta) - y
    sumSqrError = error.T @ error

    return sumSqrError / (2*m)

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha, num_iters):
    m = len(y)
    costHistory = np.zeros(num_iters)
    
    #append ones
    ones = np.ones(m)
    X = np.insert(X, 0, ones, axis=1)
    
    #Print starting cost
    cost = costJ(X, y, theta)
    print(f"Starting cost: {cost}")

    for iter in range(num_iters):
        difference = (X @ theta) - y

        # part derivative extra multiplier (From chain rule)
        gradient = (X.T @ difference) / m
        theta -= (alpha*gradient)
        
        # print cost
        cost = costJ(X, y, theta)
        costHistory[iter] = cost
        print(f"Iteration {iter+1} cost: {cost}")
        
    #plot
    plt.plot(np.arange(num_iters), costHistory)
    plt.show()

    
    return theta


data = np.genfromtxt('C:\\Users\\swguo\\VSCode Projects\\Python Projects\\Machine Learning\\Regression\\Linear Regression\\ex1data2.txt', dtype = float, delimiter = ',')

X = np.array(data[:,[0,1]])
y = np.atleast_2d(data[:,2]).T

# NORMALIZE

# axis = 0: mean of rows
mean_column = X.mean(axis=0)
mean_matrix = np.tile(mean_column, (len(X), 1))
X -= mean_matrix

std_column = X.std(axis=0)
std_matrix = np.tile(std_column, (len(X), 1))
X /= mean_matrix

# COMPUTE

theta = np.zeros((3, 1))

learningRate = 0.01
num_iters = 400

parameters = gradient_descent(X, y, theta, learningRate, num_iters)

print(parameters)


# To test other values, will need to normalize first. Subtract mean and divide by std.