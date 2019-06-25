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

""" Function for Relu """

def relu(Z):
    
    A=np.maximum(Z,0)
    activation_cache=(Z)
    
    return A,activation_cache

""" (DROPOUT) Function for Relu with Dropout """

def relu_with_dropout(Z,keep_prob):
    
    A=np.maximum(Z,0)
    D=np.random_rand(A.shape[0],A.shape[1])
    D=D<keep_prob
    A=np.multiply(A,D)
    A=A*(1/keep_prob)
    activation_cache=(Z,D)
    
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

""" (DROPOUT) Computes both Z and A for Forward Propogation with Dropout"""

def linear_activation_forward_with_dropout(A_prev,W,b,activation,keep_prob):
    
    if activation=='sigmoid':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    
    if activation=='relu':
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu_with_dropout(Z,keep_prob)
        
    cache=(linear_cache,activation_cache)
    
    return A,cache

""" Computes Activation of neuraons for all the layers """

def linear_model_forward(X,parameters,return_activations=False):
    
    send_A=[]
    caches=[]
    A=X
    L=len(parameters)//2 # // to have a integer value
    
    for i in range(1,L):
        A_prev=A
        A,cache=linear_activation_forward(A_prev,parameters['W'+str(i)],parameters['b'+str(i)],'relu')
        caches.append(cache)
        send_A.append(A)
    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    send_A.append(AL)
    
    if return_activations==False:
        return AL,caches
    else:
        return (AL,send_A)


""" (DROPOUT) Computes Activations of neurons for all the layers with Dropout """

def linear_model_forward_with_dropout(X,parameters,keep_probs):
    
    flag=0
    drop_len=len(keep_probs)
    if drop_len==1:
        flag=0
    else:
        flag=1
    caches=[]
    A=X
    L=len(parameters)//2 #?? to have a integer value
    
    for i in range(1,L):
        A_prev=A
        index=flag*i
        keep_prob=keep_probs[index]
        A,cache=linear_activation_forward_with_dropout(A_prev,parameters['W'+str(i)],parameters['b'+str(i)],'relu',keep_prob)
        caches.append(cache)
        
    AL,cache=linear_activation_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],'sigmoid')
    caches.append(cache)
    
    return(AL,caches)
        
            
            
            


""" Gives the cost of the network """

def compute_cost(AL,Y):
    
    m=Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)     
    return cost

""" (REGULARIZATIONA) Gives the cost of the network when Regularization is enabled"""

def compute_reg_cost(AL,Y,parameters,lambd):
    
    sum_weights=0
    L=len(parameters)//2
    m=Y.shape[1]
    
    for i in range(1,L+1):
        #print("W"+str(i))
        sum_weights=sum_weights+np.sum(np.square(parameters["W"+str(i)]))
        
    normal_cost=compute_cost(AL,Y)
    
    reg_cost=(lambd/(2*m))*sum_weights
    
    #print(reg_cost)
    
    final_cost=normal_cost+reg_cost
    return(final_cost)
    

""" Computes dW db dA """


def linear_backward(dZ,cache):
    
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    dW=(1/m)*np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    
    return dA_prev,dW,db


""" (REGULARIZATION) Computes dW db dA when using Regularization"""

def linear_backward_with_regularization(dZ,cache,lambd):
    
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    reg_drev=(lambd/m)*W
    #print(reg_drev)
    
    dW=(1/m)*np.dot(dZ,A_prev.T) + reg_drev 
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    #print(dW)
    return dA_prev,dW,db

""" Computes dW db dA with Dropout """

def linear_backward_with_dropout(dZ,cache,D,keep_prob):
    
    A_prev,W,b=cache
    m=A_prev.shape[1]
    
    dW=(1/m)*np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    dA_prev=np.multiply(A_prev,D)
    dA_prev=dA_prev*(1/keep_prob)
    
    return dA_prev,dW,db
    
""" Derivation for Relu """

def relu_backward(dA,activation_cache):
    #print(z)
    Z=activation_cache   
    temp,acti=relu(Z)
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


""" (REGULARIZATION) Performing all dZ,dA,db,dW with Regularization"""

def linear_activation_backward_with_regularization(dA,cache,lambd,activation):
    
    linear_cache,activation_cache=cache
    
    if activation=='sigmoid':
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward_with_regularization(dZ,linear_cache,lambd)
        
    if activation=='relu':
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward_with_regularization(dZ,linear_cache,lambd)
        
    return dA_prev,dW,db


""" Performing all dZ,dA,db,dW with Dropout """


def linear_activation_backward_with_dropout(dA,cache,activation,keep_prob):
    
    linear_cache,activation_cache=cache
    Z,D=activation_cache
    
    if activation=='sigmoid':
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    if activation=='relu':
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache,D,keep_prob)
        
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
        
        #print(i)
        current_cache=caches[i]
        dA_temp,dW_temp,db_temp=linear_activation_backward(grads['dA'+str(i+1)],current_cache,'relu')
        
        grads['dA'+str(i)]=dA_temp
        grads['dW'+str(i+1)]=dW_temp
        grads['db'+str(i+1)]=db_temp
        
    return grads


""" (REGULARIZATION) Performing BackPropogation along with Regularization """

def L_model_backward_with_regularization(AL,Y,caches,lambd):
    
    grads={}
    L=len(caches)
    
    dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
    
    current_cache=caches[L-1]
    
    grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=linear_activation_backward_with_regularization(dAL,current_cache,lambd,'sigmoid')
    
    for i in reversed(range(L-1)):
        
        #print(i)
        current_cache=caches[i]
        dA_temp,dW_temp,db_temp=linear_activation_backward_with_regularization(grads['dA'+str(i+1)],current_cache,lambd,'relu')
        
        grads['dA'+str(i)]=dA_temp
        grads['dW'+str(i+1)]=dW_temp
        grads['db'+str(i+1)]=db_temp
        
    return grads

""" (DROPOUT) Performing BackPropogation with Dropout """

def L_model_backward_with_dropout(AL,Y,caches,keep_probs):
    
    flag=0
    grads={}
    L=len(caches)
    m=AL.shape[1]
    
    if len(keep_probs)==1:
        flag==0
    else:
        flag=1
     
    dAL=-(np.divide(Y,AL)-np.divide((1-Y),(1-AL)))
    
    current_cache=caches[L-1]
    
    grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)]=linear_activation_backward_with_dropout(dAL,current_cache,'sigmoid')
    
    for i in reversed(range(L-1)):
        
        index=flag*(i+1)
        keep_prob=keep_probs[index]
        current_cache=caches[i]
        dA_temp,dW_temp,db_temp=linear_activation_backward_with_dropout(grads['dA'+str(i+1)],current_cache,'relu',keep_prob)
        
        grads['dA'+str(i)]=dA_temp
        grads['dW'+str(i+1)]=dW_temp
        grads['db'+str(i+1)]=db_temp
        
    return grads
""" Updating the parameters of W and b """

def update_parameters(parameters,grads,learning_rate):
    
    L=len(parameters)//2
    #print(grads['db1'])
    #print('end')
    for i in range(L):
        
        parameters['W'+str(i+1)]=parameters['W'+str(i+1)]-(learning_rate*grads['dW'+str(i+1)])
        parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-(learning_rate*grads['db'+str(i+1)])
        
        
    return parameters

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        