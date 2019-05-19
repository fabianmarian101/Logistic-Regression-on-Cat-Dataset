# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:15:49 2019

@author: fabian
"""
import h5py
import numpy as np
def load_dataset():
    train_dataset=h5py.File('C:/AI/train_catvnoncat.h5')
    train_set_x_orig=np.array(train_dataset['train_set_x'][:])
    train_set_y_orig=np.array(train_dataset['train_set_y'][:])

    test_dataset=h5py.File('C:/AI/test_catvnoncat.h5')
    test_set_x_orig=np.array(test_dataset['test_set_x'][:])
    test_set_y_orig=np.array(test_dataset['test_set_y'][:])

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes


    train_set_y_orig=train_set_y_orig.reshape(1,train_set_y_orig.shape[0])
    test_set_y_orig=test_set_y_orig.reshape(1,test_set_y_orig.shape[0])

    
    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes


def process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig):
    
    train_X_temp=train_set_x_orig.reshape(train_set_x_orig.shape[0]*train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],1)
    train_X=train_X_temp.reshape(train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3],train_set_x_orig.shape[0],order='F')

    test_X_temp=test_set_x_orig.reshape(test_set_x_orig.shape[0]*test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3])
    test_X=test_X_temp.reshape(test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3],test_set_x_orig.shape[0],order='F')
    
    X_train=train_X/255
    X_test=test_X/255
    
    
    Y_train=train_set_y_orig.reshape(1,train_set_y_orig.shape[1])

    Y_test=test_set_y_orig.reshape(1,test_set_y_orig.shape[1])
    
    
    return (X_train,Y_train,X_test,Y_test)

    
    
    