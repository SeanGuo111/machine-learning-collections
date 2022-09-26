
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

def sigmoid(z:np.ndarray):
    e_matrix = np.full_like(z, fill_value=np.e)
    denom = np.power(e_matrix, -1.0 * z)
    denom += 1
    return 1 / denom

def hypothesises(X: np.ndarray, theta: np.ndarray):
    z = X @ theta
    return sigmoid(z)

def costFunction(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    m = y.shape[0]
    hypos = hypothesises(X, theta)

    ifOne = -1 * y * np.log(hypos)
    ifZero = -1 * (1-y) * np.log(1 - hypos)

    cost = sum(ifOne + ifZero) / m
    return cost


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray):
    m = len(y)
    hypos = hypothesises(X, theta)

    grad = hypos - y
    grad = X.T @ grad
    grad /= m

    return grad

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha, epochs):
    m = y.shape[0]
    cost_array = [costFunction(X, y, theta)]

    for iteration in range(epochs):
        theta -= alpha * gradient(X, y, theta)
        iteration_cost = costFunction(X, y, theta)
        cost_array.append(iteration_cost)

        print(f"Iteration {iteration+1} cost: {iteration_cost[0]}")
    
    plot_cost(range(0, epochs+1), cost_array)

    return theta

def plot_cost(x_axis, y_axis):
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.ylim(0, 1)

    plt.plot(x_axis, y_axis)
    plt.show()


def plot_data(X, y, theta):
    m = len(y)
    plt.clf()
    pos = (y==1).reshape(m,1)
    neg = (y==0).reshape(m,1)
    plt.scatter(X[pos[:,0],1], X[pos[:,0],2], c="black", marker = "+")
    plt.scatter(X[neg[:,0],1], X[neg[:,0],2], c="y", marker = "o", s = 30)
    x_value = np.array([np.min(X[:,1]), np.max(X[:,1])])
    y_value = -(theta[0] + theta[1]*x_value) / theta[2]
    plt.plot(x_value, y_value, "r")
    plt.show()


# DATA
data = np.genfromtxt('C:\\Users\\swguo\\VSCode Projects\\Machine Learning\\Regression\\Logistic Regression\\ex2data1.txt', dtype = float, delimiter = ',')

X = np.array(data[:,[0,1]])
y = np.atleast_2d(data[:,2]).T
theta = np.zeros((3,1))
m = len(y)

# NORMALIZE
mean = np.mean(X, axis = 0)
std = np.std(X, axis = 0)
X = (X - mean) / std

# MODEL
X = np.insert(X, 0, np.ones(m), axis = 1)
result = gradient_descent(X, y, theta, 1, 400)
plot_data(X,y,theta)


