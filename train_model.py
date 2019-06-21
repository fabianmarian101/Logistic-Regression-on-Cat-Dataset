# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:08:17 2019

@author: Admin
"""


import numpy as np
import h5py
#import load_dataset as ld
import L_layer_neural_network as lann
import matplotlib.pyplot as plt
import visualize_ann as va
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles

#train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=ld.load_dataset()

#train_X,train_Y,test_X,test_Y=ld.process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)

#mat=scipy.io.loadmat(r"C:\personal\data.mat")

#X=mat["X"]
#y=mat["y"]
""" Create dataset and transform  the dataset """

X, y = make_circles(n_samples=200, shuffle=True,noise=0.2,factor=0.1)

train_Y=np.asarray(y)
train_X=np.asarray(X)

train_Y=train_Y.T
y_plot=train_Y
train_X=train_X.T
train_Y=train_Y.reshape(1,-1)
yy=y_plot.T
xx=train_X.T
color= ['red' if l == 0 else 'green' for l in yy]

plt.scatter(xx[:,0],xx[:,1],color=color)

""" L Layer model for training a neural network without regularization"""

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
            #print(grads)
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 20 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


""" Train a L Layer model with Regularization """

def L_layer_model_with_regularization(train_X, train_Y, layers_dims,lambd=10,learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    costs=[]
    
    #initialize weights and bias
    parameters=lann.initialize_parameters(layers_dims)
    
    
    for i in range(num_iterations):
        
        #Forward Propogation
        AL,caches=lann.linear_model_forward(train_X,parameters)
        
        #Compute Cost of the function
        cost=lann.compute_reg_cost(AL,train_Y,parameters,lambd)
        
        #Back Propogation
        grads =lann.L_model_backward_with_regularization(AL, train_Y, caches,lambd)

        #Update Weights
        parameters = lann.update_parameters(parameters, grads, learning_rate)
        
        #Printing costs
        if print_cost and i % 20 == 0:
            #print(grads)
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 20 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters



""" Predicting Y to plot on a graph """


def predict_plot(train_X,parameters):
    AL_train,caches=lann.linear_model_forward(train_X,parameters)
    AL_train[AL_train>=0.5]=1
    AL_train[AL_train<=0.5]=0
    
    return AL_train
    

layers_dims=[train_X.shape[0],4,1]# Defining the shape of the network

""" Without Regulatization """
parameters=L_layer_model(train_X, train_Y, layers_dims, learning_rate = 0.075, num_iterations = 4000, print_cost=True)# training the network without regularization

""" Wiht Regularization """
parameters2=L_layer_model_with_regularization(train_X, train_Y, layers_dims,lambd=0.7,learning_rate = 0.075, num_iterations = 14000, print_cost=True)# training the network without regularization



""" Plot Graph """

d=predict_plot(train_X,parameters).T

X1=xx[:,0].reshape(-1,1)

X2=xx[:,1].reshape(-1,1)
xig, yig = np.meshgrid(X1, X2)

y=d.reshape(-1,1)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(X1, X2)
ax.plot_trisurf(xx[:,0],xx[:,1],d[:,0])
ax.view_init(azim=120)




















