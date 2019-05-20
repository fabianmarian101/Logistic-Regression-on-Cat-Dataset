# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:54:15 2019

@author: Admin
"""

import numpy as np
import h5py 
import matplotlib.pyplot as plt


""" Initialization of parameters for N layer network """
parameters=dict()
def initialize_parameters(layer_dims):
    np.random.seed(1)
    L=len(layer_dims)
    for i in range(1,L):
        parameters['W'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])/ np.sqrt(layer_dims[i-1])
        #*0.01
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
        
    return parameters
        

""" Simple feedforward network """

def linear_forward(A,W,b):
    
    Z=np.dot(W,A)+b
    cache=(A,W,b)
    return Z,cache


def sigmoid(Z):
    
    A=1/(1+(1/np.exp(Z)))
    activation_cache=(Z)
    
    return A,activation_cache

def relu(Z):
    
    A=np.maximum(Z,0)
    activation_cache=(Z)
    
    return A,activation_cache

""" Computes both Z and A for Forward Propogation """

def linear_activation_forward(A_prev,W,b,activation):
    
    if activation=='sigmoid':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
        
    if activation=='relu':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
        
    cache=(linear_cache,activation_cache)
    
    return A,cache

""" Computes Activation of neuraons for all the layers """

def linear_model_forward(X,parameters):
    
    caches=[]
    A=X
    L=len(parameters)//2 # // to have a integer value
    
    for i in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(i)],parameters['b'+str(i)],'relu')
        caches.append(cache)
        
    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    
    return AL,caches


""" Gives the cost of the network """

def compute_cost(AL,Y):
    
    m=Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    #cost = np.squeeze(cost)     
    return cost


""" Computes dW db dA """


def linear_backward(dZ,cache):
    
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    dW=(1/m)*np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    
    return dA_prev,dW,db


""" Derivation for Relu """

def relu_backward(dA,activation_cache):
    #print(z)
    Z=activation_cache   
    temp=Z
    temp[temp>0]=1
    temp[temp<=0]=0
    dZ=dA*temp
    #print(dZ)
    return dZ

""" Derivation of Sigmoid """

def sigmoid_backward(dA,activation_cache):
    
    Z=activation_cache
    A_temp,z_temp=sigmoid(Z)
    dZ=dA*(A_temp*(1-A_temp))
    
    return dZ
    

""" Performing all dZ,dA,db,dW """

def linear_activation_backward(dA,cache,activation):
    
    linear_cache,activation_cache=cache
    
    if activation=='sigmoid':
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    if activation=='relu':
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    return dA_prev,dW,db



""" Performing BackPropogation in a single funtion """

def L_model_backward(AL,Y,caches):
    
    grads={}
    L=len(caches)
    m=AL.shape[1]
     
    dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
    
    current_cache=caches[L-1]
    
    grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=linear_activation_backward(dAL,current_cache,'sigmoid')
    
    for i in reversed(range(L-1)):
        
        
        current_cache=caches[i]
        dA_temp,dW_temp,db_temp=linear_activation_backward(grads['dA'+str(i+1)],current_cache,'relu')
        
        grads['dA'+str(i)]=dA_temp
        grads['dW'+str(i+1)]=dW_temp
        grads['db'+str(i+1)]=db_temp
        
    return grads


""" Updating the parameters of W and b """

def update_parameters(parameters,grads,learning_rate):
    
    L=len(parameters)//2
    for i in range(L):
        parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-(learning_rate*grads['dW'+str(i+1)])
        parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-(learning_rate*grads['db'+str(i+1)])
        
        
    return parameters

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        