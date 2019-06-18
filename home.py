# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:45:48 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:08:22 2019

@author: fabian
"""

import numpy as np
import h5py
#import load_dataset as ld
import L_layer_neural_network as lann
import matplotlib.pyplot as plt
import visualize_ann as va
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

#train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=ld.load_dataset()

#train_X,train_Y,test_X,test_Y=ld.process_data(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig)

mat=scipy.io.loadmat(r"C:\personal\data.mat")

X=mat["X"]
y=mat["y"]

train_Y=np.asarray(y)
train_X=np.asarray(X)

train_Y=train_Y.T
train_X=train_X.T
yy=train_Y.T
xx=train_X.T
#color= ['red' if l == 0 else 'green' for l in tt]


#plt.scatter(xx[:,0],xx[:,1],color=color)

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

def predict(test_X,test_Y,parameters,get_activations,print_option=False):
    
    AL_test,acti=lann.linear_model_forward(test_X,parameters,get_activations)
    AL_test[AL_test>=0.5]=1
    AL_test[AL_test<=0.5]=0
    if print_option==True:
        print('Predicted is %s',AL_test)
        print('Original is %s',test_Y)

    accuracy=(np.sum(test_Y-AL_test)/test_Y.shape[1])*100
    #accuracy2 = float((np.dot(test_Y,AL_test.T) + np.dot(1-test_Y,1-AL_test.T))/float(test_Y.shape[1])*100)

    #print(accuracy)
    return(acti)

def predict_train(train_X,train_Y,parameters,get_activations):
    AL_train,caches=lann.linear_model_forward(train_X,parameters,get_activations)
    AL_train[AL_train>=0.5]=1
    AL_train[AL_train<=0.5]=0
    print(AL_train)
    
    
    accuracy=(np.sum(train_Y-AL_train)/train_Y.shape[1])*100
    accuracy2 = float((np.dot(train_Y,AL_train.T) + np.dot(1-train_Y,1-AL_train.T))/float(train_Y.shape[1])*100)

    print(accuracy)
    print(accuracy2)





def predict_plot(train_X,train_Y,parameters):
    AL_train,caches=lann.linear_model_forward(train_X,parameters)
    AL_train[AL_train>=0.5]=1
    AL_train[AL_train<=0.5]=0
    
    return AL_train
    


















    
layers_dims=[train_X.shape[0],200,100,70,50,20,10,4,1]
parameters=L_layer_model(train_X, train_Y, layers_dims, learning_rate = 0.075, num_iterations = 10000, print_cost=True)







    


d=predict_plot(train_X,train_Y,parameters).T

X1=xx[:,0].reshape(-1,1)

X2=xx[:,1].reshape(-1,1)
xig, yig = np.meshgrid(X1, X2)
l.s
y=d.reshape(-1,1)
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(X1, X2)

r=np.c_[X.ravel(), Y.ravel()]

ax.plot_trisurf(xx[:,0],xx[:,1],y[:,0])
ax.view_init(azim=50)
#ax.contour3D(X,Y,y)
plt.show()


def f(x, y):
           return np.sin(np.sqrt(x ** 2 + y ** 2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)



ax = plt.axes(projection='3d')

zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'gray')







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
    labels = predict_plot(np.c_[xx.ravel(), yy.ravel()].T,y,parameters)

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels =predict_plot(X,y,parameters)
    color= ['red' if l == 0 else 'green' for l in y.T]
    ax.scatter(X[0,:], X[1,:],c=color,cmap=cmap, lw=0)

    return fig, ax


plot_decision_boundary(train_X,train_Y,parameters,cmap='RdBu')
plt.scatter(X1,X2)
















r=predict(test_X,test_Y,parameters,False)








for i in range(50):
    plt.imshow(test_set_x_orig[i])
    X1=test_X[:,i].reshape(-1,1)
    Y1=test_Y[0,i].reshape(-1,1)
    get_activations=True
    A_t=predict(X1,Y1,parameters,get_activations)

    network = va.DrawNN( [10,7,1] )
    network.draw(A_t)
    A_t=predict(X1,Y1,parameters,get_activations,True)


100