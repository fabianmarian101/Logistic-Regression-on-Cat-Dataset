# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:08:22 2019

@author: fabian
"""

import numpy as np
import h5py
import load_dataset as ld
import L_layer_neural_network as lann
import matplotlib.pyplot as plt

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=ld.load_dataset()

train_X,train_Y,test_X,test_Y=ld.process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)


def L_layer_model(train_X, train_Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    costs=[]
    
    #initialize weights and bias
    parameters=lann.initialize_parameters(layers_dims)
    
    
    for i in range(num_iterations):
        
        #Forward Propogation
        AL,caches=lann.linear_model_forward(train_X,parameters)
        
        #Compute Cost of the function
        cost=lann.compute_cost(AL,train_Y)
        
        #Back Propogation
        grads =lann.L_model_backward(AL, train_Y, caches)

        #Update Weights
        parameters = lann.update_parameters(parameters, grads, learning_rate)
        
        #Printing costs
        if print_cost and i % 20 == 0:
            print(grads)
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 20 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def predict(test_X,test_Y,parameters):
    
    AL_test,caches=lann.linear_model_forward(test_X,parameters)
    AL_test[AL_test>=0.5]=1
    AL_test[AL_test<=0.5]=0
    
    accuracy=(np.sum(test_Y-AL_test)/test_Y.shape[1])*100
    accuracy2 = float((np.dot(test_Y,AL_test.T) + np.dot(1-test_Y,1-AL_test.T))/float(test_Y.shape[1])*100)

    print(accuracy)
    print(accuracy2)

def predict_train(train_X,train_Y,parameters):
    AL_train,caches=lann.linear_model_forward(train_X,parameters)
    AL_train[AL_train>=0.5]=1
    AL_train[AL_train<=0.5]=0
    
    accuracy=(np.sum(train_Y-AL_train)/train_Y.shape[1])*100
    accuracy2 = float((np.dot(train_Y,AL_train.T) + np.dot(1-train_Y,1-AL_train.T))/float(train_Y.shape[1])*100)

    print(accuracy)
    print(accuracy2)
    
layers_dims=[train_X.shape[0],7,5,1]
parameters=L_layer_model(train_X, train_Y, layers_dims, learning_rate = 0.75, num_iterations = 1000, print_cost=True)
predict(test_X,test_Y,parameters)
100