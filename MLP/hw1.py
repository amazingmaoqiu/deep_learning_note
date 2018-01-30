import sys
import argparse
import numpy as np
import math

def load_data(filePath):
    X = []
    Y = []
    X = np.loadtxt(filePath, delimiter = ',', usecols = list(range(784)))
    Y = np.loadtxt(filePath, delimiter = ',', usecols = [784])
    return X, Y

#PROBLEM 1    
def update_weights_perceptron(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i,int(Y[i])] = 1.0
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.dot(X, weights[-1]) + bias[-1]

    output = np.exp(a) / np.sum(np.exp(a), axis = 1, keepdims = True)
    cache['a'].append(a)
    cache['z'].append(output)
    # loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat

    dw = np.dot(cache['z'][-2].transpose(), da) / float(Y.shape[0])
    # dw = np.dot(X.transpose(), da) / float(Y.shape[0])
    db = np.sum(da,axis = 0) / float(Y.shape[0])

    updated_weights[-1] -= lr*dw 
    updated_bias[-1]    -= lr*db 
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
        Y_hat[i,int(Y[i])] = 1.0
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.dot(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / np.repeat(np.sum(np.exp(a), axis = 1),10).reshape(-1,10)
    cache['a'].append(a)
    cache['z'].append(output)
    # loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] -= lr*dw / Y.shape[0]
    updated_bias[-1]    -= lr*db / Y.shape[0]
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw / Y.shape[0]
        # updated_weights[i-1] = np.zeros(weights[i-1].shape) / Y.shape[0]
        updated_bias[i-1]    += -1.0*lr*db / Y.shape[0]
    return updated_weights, updated_bias

#PROBLEM 3
# def update_weights_single_layer_mean(X, Y, weights, bias, lr):
#     #INSERT YOUR CODE HERE
    
#     return updated_weights, updated_bias

#PROBLEM 4
def update_weights_double_layer(X, Y, weights, bias, lr):
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i,int(Y[i])] = 1.0
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        z = 1.0 / (1.0 + np.exp(-1.0 * a))
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.dot(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / np.repeat(np.sum(np.exp(a), axis = 1),10).reshape(-1,10)
    cache['a'].append(a)
    cache['z'].append(output)
    # loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] -= lr*dw / Y.shape[0]
    updated_bias[-1]    -= lr*db / Y.shape[0]
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw / Y.shape[0]
        # updated_weights[i-1] = np.zeros(weights[i-1].shape) / Y.shape[0]
        updated_bias[i-1]    += -1.0*lr*db / Y.shape[0]
    return updated_weights, updated_bias

#PROBLEM 5
def update_weights_double_layer_batch(X, Y, weights, bias, lr, activation):
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
        print(str(i_batch) + 'th batch completed.')
            
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
        print(str(num_batch) + 'th batch completed.')
    updated_weights, updated_bias = weights, bias
    # print(Y_hat)
    return updated_weights, updated_bias

#PROBLEM 6
def update_weights_double_layer_act(X, Y, weights, bias, lr, activation):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : 1.0 / (1.0 + np.exp(-1.0 * x))
        back_activate     = lambda x, z : (1.0 - z)*z
    if activation == 'tanh':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.tanh(x)
        back_activate     = lambda x, z : 1.0 - z*z
    if activation == 'relu':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.maximum(0,x)
        back_activate     = lambda x, z : (x > 0).astype(float)
    #INSERT YOUR CODE HERE
    cache = {'a':[],'z':[]}
    z = X
    Y_hat = np.zeros([Y.shape[0],10])
    for i in range(Y.shape[0]):
        Y_hat[i,int(Y[i])] = 1.0
    cache['z'].append(z)
    for i in range(len(weights)-1):
        a = np.matmul(z, weights[i]) + bias[i]
        # z = 1.0 / (1.0 + np.exp(-1.0 * a))
        z = activate_function(a)
        cache['a'].append(a)
        cache['z'].append(z)
    a = np.dot(z, weights[len(weights)-1]) + bias[len(weights)-1]
    output = np.exp(a) / np.sum(np.exp(a), axis = 1, keepdims = True)
    # output = np.exp(a) / np.repeat(np.sum(np.exp(a), axis = 1),10).reshape(-1,10)
    # output = activate_function(a)
    cache['a'].append(a)
    cache['z'].append(output)
    # loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
    #backpropagation
    updated_weights, updated_bias = weights, bias
    da = output - Y_hat
    # dout = Y_hat / output
    # da = dout * back_activate(cache['a'][-1], cache['z'][-1])
    dw = np.dot(cache['z'][-2].T, da)
    db = np.sum(da,axis = 0)
    updated_weights[-1] -= lr*dw / Y.shape[0]
    updated_bias[-1]    -= lr*db / Y.shape[0]
    for i in range(len(weights)-1,0,-1):
        dz = np.dot(da,weights[i].T)
        # da = dz*(1.0 - cache['z'][i])*cache['z'][i]
        da = dz*back_activate(cache['a'][i-1], cache['z'][i])
        dw = np.dot(cache['z'][i-1].T,da)
        db = np.sum(da,axis=0)
        updated_weights[i-1] += -1.0*lr*dw / Y.shape[0]
        # updated_weights[i-1] = np.zeros(weights[i-1].shape) / Y.shape[0]
        updated_bias[i-1]    += -1.0*lr*db / Y.shape[0]
    return updated_weights, updated_bias

#PROBLEM 7
def update_weights_double_layer_act_mom(X, Y, weights, bias, lr, activation, momentum, epochs):
    #INSERT YOUR CODE HERE
    if activation == 'sigmoid':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : 1.0 / (1.0 + np.exp(-1.0 * x))
        back_activate     = lambda x, z : (1.0 - z)*z
    if activation == 'tanh':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.tanh(x)
        back_activate     = lambda x, z : 1.0 - z*z
    if activation == 'relu':
        #INSERT YOUR CODE HERE
        activate_function = lambda x : np.maximum(0,x)
        back_activate     = lambda x, z : (x > 0).astype(float)
    #INSERT YOUR CODE HERE
    updated_weights, updated_bias = weights, bias
    cache = {'a':[],'z':[]}

    vw = []
    vb = []
    for i in range(len(weights)):
        vw.append(0.0)
        vb.append(0.0)

    z = X

    for idx_epoch in range(epochs):
        z = X
        Y_hat = np.zeros([Y.shape[0],10])
        for i in range(Y.shape[0]):
            Y_hat[i,int(Y[i])] = 1.0
        cache['z'].append(z)
        for i in range(len(weights)-1):
            a = np.matmul(z, weights[i]) + bias[i]
            # z = 1.0 / (1.0 + np.exp(-1.0 * a))
            z = activate_function(a)
            cache['a'].append(a)
            cache['z'].append(z)
        a = np.dot(z, weights[len(weights)-1]) + bias[len(weights)-1]
        output = np.exp(a) / np.sum(np.exp(a), axis = 1, keepdims = True)
        # output = np.exp(a) / np.repeat(np.sum(np.exp(a), axis = 1),10).reshape(-1,10)
        # output = activate_function(a)
        cache['a'].append(a)
        cache['z'].append(output)
        # loss = -1.0 * (Y_hat*np.log(output) + (1.0 - Y_hat)*np.log(1.0 - output))
        #backpropagation
        
        da = output - Y_hat
        dw = np.dot(cache['z'][-2].T, da)
        db = np.sum(da,axis = 0)
        vw[-1] = momentum*vw[-1] + (1.0 - momentum)*dw
        vb[-1] = momentum*vb[-1] + (1.0 - momentum)*db
        updated_weights[-1] -= lr*vw[-1] / Y.shape[0]
        updated_bias[-1]    -= lr*vb[-1] / Y.shape[0]
        for i in range(len(weights)-1,0,-1):
            dz = np.dot(da,weights[i].T)
            # da = dz*(1.0 - cache['z'][i])*cache['z'][i]
            da = dz*back_activate(cache['a'][i-1], cache['z'][i])
            dw = np.dot(cache['z'][i-1].T,da)
            db = np.sum(da,axis=0)
            vw[i-1] = momentum*vw[i-1] + (1.0 - momentum)*dw
            vb[i-1] = momentum*vb[i-1] + (1.0 - momentum)*db
            updated_weights[i-1] += -1.0*lr*vw[i-1] / Y.shape[0]
            # updated_weights[i-1] = np.zeros(weights[i-1].shape) / Y.shape[0]
            updated_bias[i-1]    += -1.0*lr*vb[i-1] / Y.shape[0]
        print('epoch ' + str(idx_epoch) + ' completed.')
    return updated_weights, updated_bias
    
# def main():
#     X, Y = load_data("./digitstrain.txt")
#     print(Y)
#     weights = []
#     bias = []
#     batch_size = 333
#     weights.append(np.random.rand(784,10))
#     bias.append(np.random.rand(10))
#     # weights.append(np.random.rand(100,100))
#     # bias.append(np.random.rand(100))
#     # weights.append(np.random.rand(100,10))
#     # bias.append(np.random.rand(10))
#     learning_rate = 1e-3
#     weights, bias = update_weights_perceptron(X,Y,weights,bias,learning_rate)
# # #    weights, bias = update_weights_double_layer(X, Y, weights, bias, learning_rate)
# #     weights,bias = update_weights_double_layer_batch(X, Y, weights, bias, learning_rate, batch_size)
#     # print(weights[-1].shape)
#     print(type(X))
#     print(Y.shape[0])
#     # print(weights.shape)
#     # print(bias.shape)
    
    
# if __name__ == "__main__":
#     main()
