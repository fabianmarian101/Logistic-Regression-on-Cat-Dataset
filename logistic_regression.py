# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:38:29 2019

@author: Admin
"""

import numpy as np 
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
np.random.seed(1) # set a seed so that the results are consistent
def load_dataset():
    train_dataset=h5py.File('C:/personal/train_catvnoncat.h5')
    train_set_x_orig=np.array(train_dataset['train_set_x'][:])
    train_set_y_orig=np.array(train_dataset['train_set_y'][:])

    test_dataset=h5py.File('C:/personal/test_catvnoncat.h5')
    test_set_x_orig=np.array(test_dataset['test_set_x'][:])
    test_set_y_orig=np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes


    train_set_y_orig=train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    test_set_y_orig=test_set_y_orig.reshape(1,test_set_y_orig.shape[0])

    
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes




train_set_x_orig,train_set_y,test_set_x_orig,test_set_y,classes=load_dataset()

#flatting the pixels into a  single vector
train_X=train_set_x_orig.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0])
test_X=test_set_x_orig.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0])

#Standardise the dataset

X_train=train_X/255
X_test=test_X/255

#Reshaping Target values for both train and test labels

Y_train=train_set_y.reshape(1,train_set_y.shape[1])

Y_test=test_set_y.reshape(1,test_set_y.shape[1])


m_train=train_set_x_orig.shape[0]
m_test=test_set_x_orig.shpae[0]





def intialize_parameters(dims):
    
    w=np.random.randn(dims,1)*0.01 
    b=0
    
    return w,b

def sigmoid(z):
    
    return 1/(1+(1/np.exp(z)))


def propogate(w,b,X,Y):
    
    m=X.shape[1]
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    cost=-(1/m)*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))
    
    dw=(1/m)*np.dot(X,(A-Y).T)
    
    db=(1/m)*np.sum(A-Y)
    
    grads={'dw':dw,'db':db}
    
    return grads,cost


def optimize(w,b,X,Y,learning_rate,num_iterations,print_cost=False):
    
    costs=[]
    for i in range(num_iterations):
        
       grads,cost= propogate(w,b,X,Y)
       
       if i%10==0:
           costs.append(cost)
           
       dw=grads['dw']
       db=grads['db']
       
       w=w-learning_rate*dw
       b=b-learning_rate*db
       
       if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            
            
    params={'w':w,'b':b}
    
    return params,cost


def predict(w,b,X):
    
    m=X.shape[1]
    w=w.reshape(X.shape[0],1)
   # Y_prediction=np.zeros((1,m))
    
    A=sigmoid(np.dot(w.T,X)+b)
    
    A[A>0.5]=1
    
    A[A<0.5]=0
    
    return A
    
    







