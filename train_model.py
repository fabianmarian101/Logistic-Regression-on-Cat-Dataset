# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:08:17 2019

@author: Admin
"""


import numpy as np
import h5py
import load_dataset as ld
import L_layer_neural_network as lann
import matplotlib.pyplot as plt
#import visualize_ann as va
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
import math

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=ld.load_dataset()

train_X,train_Y,test_X,test_Y=ld.process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)


#train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=ld.load_dataset()

#train_X,train_Y,test_X,test_Y=ld.process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)

#mat=scipy.io.loadmat(r"C:\personal\data.mat")

#X=mat["X"]
#y=mat["y"]
""" Create dataset and transform  the dataset """

#X, y = make_circles(n_samples=200, shuffle=True,noise=0.2,factor=0.12)
#mat=scipy.io.loadmat(r"C:\personal\data.mat")
"""
X, y = make_moons(n_samples=200, shuffle=True,noise=0.2)

#X=mat["X"]
#y=mat["y"]

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
plt.show()

#t_X,t_Y=make_circles(n_samples=200, shuffle=True,noise=0.17,factor=0.12)

t_X,t_Y=make_moons(n_samples=200, shuffle=True,noise=0.17)
test_Y=np.asarray(t_Y)
test_X=np.asarray(t_X)

test_Y=test_Y.T
y_plot2=test_Y
test_X=test_X.T
test_Y=test_Y.reshape(1,-1)
yy2=y_plot2.T
xx2=test_X.T
color= ['red' if l == 0 else 'green' for l in yy2]

plt.scatter(xx2[:,0],xx2[:,1],color=color)
plt.show()
"""

""" Create Random mini Batches """
def random_mini_batches(X,Y,mini_batch,seed):
    
    np.random.seed(seed)
    m=X.shape[1]
    mini_batch_X=[]
    mini_batch_Y=[]
    
    permutation=list(np.random.permutation(m))
    
    shuffled_X=X[:,permutation]
    shuffled_Y=Y[:,permutation].reshape((1,m))
    
    num_of_mini_batches=math.floor(m/mini_batch)
    
    for i in range(num_of_mini_batches):
        mini_batch_X.append(shuffled_X[:,i*mini_batch:(i+1)*mini_batch])
        mini_batch_Y.append(shuffled_Y[:,i*mini_batch:(i+1)*mini_batch])
        
    if m%mini_batch!=0:
        mini_batch_X.append(shuffled_X[:,mini_batch*num_of_mini_batches:m])
        mini_batch_Y.append(shuffled_Y[:,mini_batch*num_of_mini_batches:m])
        
    return (mini_batch_X,mini_batch_Y)

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

def L_layer_model_with_regularization(train_X, train_Y, layers_dims,lambd=0,learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    costs=[]
    print(lambd)
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

""" Training a L nmodel with Dropout """
catch=()
def L_layer_model_with_dropout(train_X, train_Y, layers_dims,keep_probs, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    global catch
    np.random.seed(1)
    costs=[]
    
    #initialize weights and bias
    parameters=lann.initialize_parameters(layers_dims)
    
    
    for i in range(num_iterations):
        
        #Forward Propogation
        AL,caches,D_collect=lann.linear_model_forward_with_dropout(train_X,parameters,keep_probs)
        catch=caches
        #print(caches)
        #Compute Cost of the function
        cost=lann.compute_cost(AL,train_Y)
        
        #Back Propogation
        grads =lann.L_model_backward_with_dropout(AL, train_Y, caches,D_collect,keep_probs)

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


""" Training a L Layer model with mini batches """

def L_layer_model_with_batch(train_X, train_Y, layers_dims,mini_batch,beta,learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    
    np.random.seed(1)
    costs=[]
    
    X_mini_batch,Y_min_batch=random_mini_batches(train_X,train_Y,mini_batch)
    #initialize weights and bias
    #print(X_mini_batch[0].shape)
    
    parameters=lann.initialize_parameters(layers_dims)
    v=lann.initialize_velocity(parameters)
    
    num_of_mini_batch=len(X_mini_batch)
    
    for i in range(num_iterations):
        for j in range(num_of_mini_batch):  
        #Forward Propogation
            
            AL,caches=lann.linear_model_forward(X_mini_batch[j],parameters)
        
        #Compute Cost of the function
            cost=lann.compute_cost(AL,Y_min_batch[j])
        
        #Back Propogation
            grads =lann.L_model_backward(AL, Y_min_batch[j], caches)

        #Update Weights
            #parameters,v = lann.update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            lann.update_parameters(parameters, grads, learning_rate)
        #Printing costs
        #if print_cost and j % 4 == 0:
            #print(grads)
        print ("Cost after iteration %i: %f" %(i, cost))
        #if print_cost and i % 4 == 0:
        costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


""" L  layer model with all optimization """

def model(train_X,train_Y,layers_dims,optimizer,learning_rate=0.0001,mini_batch=8,beta=0.9,beta1=0.9,beta2=0.999,epsilon=1e-8,num_iterations=10000,print_cost=True):
    
    seed=10
    costs=[]
    t=0
    
    parameters=lann.initialize_parameters(layers_dims)
    
    if optimizer=='momentum':
        v=lann.initialize_velocity(parameters)
    elif optimizer=='adam':
        v,s=lann.initialize_adam(parameters)
        
    for i in range(num_iterations):
        seed=seed+1
        X_mini_batch,Y_mini_batch=random_mini_batches(train_X,train_Y,mini_batch,seed)
        num_of_batch=len(X_mini_batch)
        
        for j in range(num_of_batch):
            
            #Forward Propogation
            AL,caches=lann.linear_model_forward(X_mini_batch[j],parameters)
            #Compute Cost of the function
            cost=lann.compute_cost(AL,Y_mini_batch[j])
            #Back Propogation
            grads =lann.L_model_backward(AL, Y_mini_batch[j], caches)
            
            if optimizer=='gd':
                parameters= lann.update_parameters(parameters,grads,learning_rate)
            elif optimizer=='momentum':
                parameters,v=lann.update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer=='adam':
                t=t+1
                parameters=lann.adam_optimiser(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
                
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
        print(i)
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters








""" Predicting Y to plot on a graph """



def predict_plot2(train_X,train_Y,parameters):
    AL_train,caches=lann.linear_model_forward(train_X,parameters)
    AL_train[AL_train>=0.5]=1
    AL_train[AL_train<=0.5]=0
    
    return AL_train
    
    

layers_dims=[train_X.shape[0],12,7,5,1]# Defining the shape of the network

""" With min Batch """
parameters4=L_layer_model_with_batch(train_X,train_Y, layers_dims,10,0.9,learning_rate = 0.075, num_iterations = 1050, print_cost=True)

""" Without Regulatization """
parameters=L_layer_model(train_X, train_Y, layers_dims, learning_rate = 0.0075, num_iterations = 5000, print_cost=True)# training the network without regularization

""" With Regularization """
parameters2=L_layer_model_with_regularization(train_X, train_Y, layers_dims,lambd=2,learning_rate = 0.075, num_iterations = 5000, print_cost=True)# training the network without regularization


""" With Dropout """
keep_probs=[0.5,0.5,0.5]
parameters3=L_layer_model_with_dropout(train_X, train_Y, layers_dims,keep_probs,learning_rate = 0.075, num_iterations =3000, print_cost=True)# training the network without regularization


""" with optimiser """

parameters5=model(train_X,train_Y,layers_dims,optimizer='adam',learning_rate=0.0007,num_iterations=1000)





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







def plot_decision_boundary(X, y, parameters, steps=1000, cmap='Paired'):
    cmap = plt.get_cmap(cmap)

    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = predict_plot2(np.c_[xx.ravel(), yy.ravel()].T,y,parameters)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    #train_labels =predict_plot(X,parameters)
    X_disp=X.T
    color= ['red' if l == 0 else 'green' for l in y.T]
    ax.scatter(X_disp[0,:], X_disp[1,:],c=color,cmap=cmap, lw=0)
    #ax.set_yticks((-0.5,0.8))
    #ax.set_xticks((-0.5,0.5))

    return fig, ax


plot_decision_boundary(train_X.T,train_Y,parameters,cmap='RdBu')

plot_decision_boundary(train_X.T,train_Y,parameters2,cmap='BrBG')

plot_decision_boundary(train_X.T,train_Y,parameters3,cmap='BrBG')

plot_decision_boundary(train_X.T,train_Y,parameters4,cmap='BrBG')


def predict(test_X,test_Y,parameters):
    
    AL_test,acti=lann.linear_model_forward(test_X,parameters)
    AL_test[AL_test>=0.5]=1
    AL_test[AL_test<=0.5]=0
    #accuracy=(np.sum(test_Y-AL_test)/test_Y.shape[1])*100
    accuracy2 = float((np.dot(test_Y,AL_test.T) + np.dot(1-test_Y,1-AL_test.T))/float(test_Y.shape[1])*100)
    
    #print("Accuracy is "+str(accuracy))
    
    print("Accuracy2 is "+str(accuracy2))
    #print(accuracy)
    #return(acti)


predict(test_X,test_Y,parameters)
plot_decision_boundary(test_X.T,test_Y,parameters,cmap="BrBG")

predict(test_X,test_Y,parameters2)
plot_decision_boundary(test_X.T,test_Y,parameters2,cmap="BrBG")

predict(test_X,test_Y,parameters3)
plot_decision_boundary(test_X.T,test_Y,parameters3,cmap="BrBG")



predict(test_X,test_Y,parameters4)
plot_decision_boundary(test_X.T,test_Y,parameters4,cmap="BrBG")

predict(test_X,test_Y,parameters5)
plot_decision_boundary(test_X.T,test_Y,parameters5,cmap="BrBG")







