import numpy as np
import pandas as pd
import matplotlib as plt

def hypothesis(p1, p2, x):
    return p1 + (p2*x)

def g_descent(batch, p1, p2, learning_rate):
    
    p1_partderiv = 1
    p2_partderiv = 1

    while (not abs(p1_partderiv) <= 0.001) or (not abs(p2_partderiv) <= 0.001):
        p1_partderiv = 0
        p2_partderiv = 0

        #Cost
        for coordinate in batch:
            x_coord = coordinate[0]
            y_coord = coordinate[1]
            difference = (hypothesis(p1, p2, x_coord) - y_coord)
            p1_partderiv += difference
            p2_partderiv += (difference * x_coord)
    
        #Smaller and learning rate
        p1_partderiv /= len(batch)
        p1_partderiv *= learning_rate
        p2_partderiv /= len(batch)
        p2_partderiv *= learning_rate
        
        p1 -= p1_partderiv
        p2 -= p2_partderiv
        print(p1)
        print(p2)
    
    print()
    print("Final result:")
    print(f"{p1} + {p2}x")

def normal_eq(batch):
    #Formula: projection from perfect line down to playing field matrix of y values.
    arrA = []
    vectB = []
    for coord in batch:
        arrA.append([1, coord[0]])
        vectB.append(coord[1])
    arrAT = []


batch = [(1,1), (2,2), (3,2)]
g_descent(batch, 0, 0, 0.01)

