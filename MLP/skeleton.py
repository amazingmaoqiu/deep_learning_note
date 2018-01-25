import sys
import argparse
import numpy as np
import math
import pdb
import random
import pickle

def load_data(filePath):
    X = []
    Y = []
    file = open(filePath,'r')
    input = file.readlines()
    for line in input:
        line_list = line.split(',')
        Y.append(line_list[-1])
        X.append([float(i) for i in line_list[0:-1]])
    X.pop()
    Y.pop()
    X = np.array(X)
    Y = np.array([int(i) for i in Y])
    return X, Y

#PROBLEM 1    
def update_weights_perceptron(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i][Y[i]] = 1
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / (1.0 + np.exp(a))
    cache['a'].append(a)
    cache['z'].append(output)
    loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] += -1.0*lr*dw
    updated_bias[-1]    += -1.0*lr*db
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw
        updated_bias[i-1]    += -1.0*lr*db
    return updated_weights, updated_bias

#PROBLEM 2
def update_weights_single_layer(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i][Y[i]] = 1
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / (1.0 + np.exp(a))
    cache['a'].append(a)
    cache['z'].append(output)
    loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] += -1.0*lr*dw
    updated_bias[-1]    += -1.0*lr*db
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw
        updated_bias[i-1]    += -1.0*lr*db
    return updated_weights, updated_bias

#PROBLEM 3
def update_weights_single_layer_mean(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    
    return updated_weights, updated_bias

#PROBLEM 4
def update_weights_double_layer(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i][Y[i]] = 1
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / (1.0 + np.exp(a))
    cache['a'].append(a)
    cache['z'].append(output)
    loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] += -1.0*lr*dw
    updated_bias[-1]    += -1.0*lr*db
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw
        updated_bias[i-1]    += -1.0*lr*db
    return updated_weights, updated_bias

#PROBLEM 5
def update_weights_double_layer_batch(X, Y, weights, bias, lr, batch_size):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    num_batch = Y.shape[0] // batch_size
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i][Y[i]] = 1
    for i_batch in range(num_batch):
        cache['a'].clear()
        cache['z'].clear()
        z = X[i_batch*batch_size:(i_batch+1)*batch_size,:]
        cache['z'].append(z)
        for i in range(len(weights)-1):
            a = np.matmul(z, weights[i]) + bias[i]
            z = 1.0 / (1.0 + np.exp(-1.0 * a))
            cache['a'].append(a)
            cache['z'].append(z)
        a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
        output = np.exp(a) / (1.0 + np.exp(a))
        cache['a'].append(a)
        cache['z'].append(output)
        loss = -1.0 * (Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:]*np.log(output) + (1.0 - Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:])*np.log(1.0 - output))
        #backpropagation
        da = output - Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:]
        dw = np.dot(cache['z'][-2].T, da)
        db = np.sum(da,axis = 0)
        weights[-1] += -1.0*lr*dw
        bias[-1]    += -1.0*lr*db
        for i in range(len(weights)-1,0,-1):
            dz = np.dot(da,weights[i].T)
            print(cache['z'][i].shape)
            print(dz.shape)
            da = dz*(1.0 - cache['z'][i])*cache['z'][i]
            dw = np.dot(cache['z'][i-1].T,da)
            db = np.sum(da,axis=0)
            weights[i-1] += -1.0*lr*dw
            bias[i-1]    += -1.0*lr*db
        print(i_batch)
            
#     last batch
    if(num_batch*batch_size < Y.shape[0]):
        cache['a'].clear()
        cache['z'].clear()
        z = X[num_batch*batch_size:,:]
        print(z.shape)
        cache['z'].append(z)
        for i in range(len(weights)-1):
            a = np.matmul(z, weights[i]) + bias[i]
            z = 1.0 / (1.0 + np.exp(-1.0 * a))
            cache['a'].append(a)
            cache['z'].append(z)
        a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
        print(z.shape)
        output = np.exp(a) / (1.0 + np.exp(a))
        cache['a'].append(a)
        cache['z'].append(output)
        loss = -1.0 * (Y_hat[num_batch*batch_size:,:]*np.log(output) + (1.0 - Y_hat[num_batch*batch_size:,:])*np.log(1.0 - output))
        #backpropagation
        da = output - Y_hat[num_batch*batch_size:,:]
        dw = np.dot(cache['z'][-2].T, da)
        db = np.sum(da,axis = 0)
        weights[-1] += -1.0*lr*dw
        bias[-1]    += -1.0*lr*db
        for i in range(len(weights)-1,0,-1):
            dz = np.dot(da,weights[i].T)
            print(cache['z'][i].shape)
            print(da.shape)
            da = dz*(1.0 - cache['z'][i])*cache['z'][i]
            dw = np.dot(cache['z'][i-1].T,da)
            db = np.sum(da,axis=0)
            weights[i-1] += -1.0*lr*dw
            bias[i-1]    += -1.0*lr*db
        print(num_batch)
    updated_weights, updated_bias = weights, bias
    print(Y_hat)
    return updated_weights, updated_bias

#PROBLEM 6
def update_weights_double_layer_batch_act(X, Y, weights, bias, lr, batch_size, activation):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : 1.0 / (1.0 + np.exp(-1.0 * x))
    if activation == 'tanh':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.tanh(x)
    if activation == 'relu':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.maximum(0,x)
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    num_batch = Y.shape[0] // batch_size
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i][Y[i]] = 1
    for i_batch in range(num_batch):
        cache['a'].clear()
        cache['z'].clear()
        z = X[i_batch*batch_size:(i_batch+1)*batch_size,:]
        cache['z'].append(z)
        for i in range(len(weights)-1):
            a = np.matmul(z, weights[i]) + bias[i]
            z = activate_function(a)
            cache['a'].append(a)
            cache['z'].append(z)
        a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
#         output = np.exp(a) / (1.0 + np.exp(a))
        output = activate_function(a)
        cache['a'].append(a)
        cache['z'].append(output)
        loss = -1.0 * (Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:]*np.log(output) + (1.0 - Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:])*np.log(1.0 - output))
        #backpropagation
        da = output - Y_hat[i_batch*batch_size:(i_batch+1)*batch_size,:]
        dw = np.dot(cache['z'][-2].T, da)
        db = np.sum(da,axis = 0)
        weights[-1] += -1.0*lr*dw
        bias[-1]    += -1.0*lr*db
        for i in range(len(weights)-1,0,-1):
            dz = np.dot(da,weights[i].T)
            print(cache['z'][i].shape)
            print(dz.shape)
            da = dz*(1.0 - cache['z'][i])*cache['z'][i]
            dw = np.dot(cache['z'][i-1].T,da)
            db = np.sum(da,axis=0)
            weights[i-1] += -1.0*lr*dw
            bias[i-1]    += -1.0*lr*db
        print(i_batch)
            
#     last batch
    if(num_batch*batch_size < Y.shape[0]):
        cache['a'].clear()
        cache['z'].clear()
        z = X[num_batch*batch_size:,:]
        print(z.shape)
        cache['z'].append(z)
        for i in range(len(weights)-1):
            a = np.matmul(z, weights[i]) + bias[i]
            z = activate_function(a)
            cache['a'].append(a)
            cache['z'].append(z)
        a = np.matmul(z, weights[len(weights)-1]) + bias[len(weights)-1]
        print(z.shape)
#         output = np.exp(a) / (1.0 + np.exp(a))
        output = activate_function(a)
        cache['a'].append(a)
        cache['z'].append(output)
        loss = -1.0 * (Y_hat[num_batch*batch_size:,:]*np.log(output) + (1.0 - Y_hat[num_batch*batch_size:,:])*np.log(1.0 - output))
        #backpropagation
        da = output - Y_hat[num_batch*batch_size:,:]
        dw = np.dot(cache['z'][-2].T, da)
        db = np.sum(da,axis = 0)
        weights[-1] += -1.0*lr*dw
        bias[-1]    += -1.0*lr*db
        for i in range(len(weights)-1,0,-1):
            dz = np.dot(da,weights[i].T)
            print(cache['z'][i].shape)
            print(da.shape)
            da = dz*(1.0 - cache['z'][i])*cache['z'][i]
            dw = np.dot(cache['z'][i-1].T,da)
            db = np.sum(da,axis=0)
            weights[i-1] += -1.0*lr*dw
            bias[i-1]    += -1.0*lr*db
        print(num_batch)
    updated_weights, updated_bias = weights, bias
    return updated_weights, updated_bias

#PROBLEM 7
def update_weights_double_layer_batch_act_mom(X, Y, weights, bias, lr, batch_size, activation, momentum):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        #INSERT YOUR CODE HERE
        print(1)
    if activation == 'tanh':
        #INSERT YOUR CODE HERE
        print(1)
    if activation == 'relu':
        #INSERT YOUR CODE HERE
        print(1)
    #INSERT YOUR CODE HERE
    return updated_weights, updated_bias
    
def main():
    X, Y = load_data("./digitstrain.txt")
    weights = []
    bias = []
    batch_size = 333
    weights.append(np.random.rand(784,100))
    bias.append(np.random.rand(100))
    weights.append(np.random.rand(100,100))
    bias.append(np.random.rand(100))
    weights.append(np.random.rand(100,10))
    bias.append(np.random.rand(10))
    learning_rate = 1e-3
#     weights, bias = update_weights_perceptron(X,Y,weights,bias,learning_rate)
#    weights, bias = update_weights_double_layer(X, Y, weights, bias, learning_rate)
    weights,bias = update_weights_double_layer_batch(X, Y, weights, bias, learning_rate, batch_size)
    print(weights[-1].shape)
    
    
if __name__ == "__main__":
    main()
