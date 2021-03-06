# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 22:53:37 2022

@author: Julien Nyambal
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def predict(a):
    h = a[0]*w_1 + a[1]*w_2 + w_b
    return sigmoid(h)

w_1 = 0.23

w_2 = 0.12

w_b = 0.01

lr = 0.25

input_set = np.array([[1,0,0],
                      [1,0,1],
                      [1,1,0],
                      [1,1,1]])

labels = np.array([0, 1, 1, 1])

error_lst = [10]
error = 0.1
while error > 0.000000001:
    for x, y, in zip(input_set, labels):
    
        h = x[1]*w_1 + x[2]*w_2 + x[0]*w_b
        y_hat = sigmoid(h)
        
        # Gradient bias
        delta_w_b = x[0] * sigmoid_prime(h) * (y_hat - y)
        # Gradients weights
        delta_w_1 = x[1] * sigmoid_prime(h) * (y_hat - y)
        delta_w_2 = x[2] * sigmoid_prime(h) * (y_hat - y)
    
        w_b = w_b - lr * delta_w_b
        w_1 = w_1 - lr * delta_w_1
        w_2 = w_2 - lr * delta_w_2
        
        weights =  np.array([[w_b, w_1, w_2]])
        
        error_lst.append(sum((sigmoid(np.matmul(input_set, weights.T))[:, 0] - labels)**2))
        error = error_lst[-2] - error_lst[-1]
        
    print(error)



print("")
print(predict([0,0]))
print(predict([0,1]))
print(predict([1,0]))
print(predict([1,1]))
    


