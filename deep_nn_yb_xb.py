# standard imports
import numpy as np
import matplotlib.pyplot as plt
import math
import random

# data handling imports
import pandas as pd
import csv

# tensorflow imports
import h5py
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
from data_augmentation_and_transformation import augment_scalar
from data_augmentation_and_transformation import binary_to_cartes

# Creates Random Minibatches (credit: Andrew Ng and deeplearning.ai)
def random_mini_batches(X, Y, mini_batch_size = 180, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# initializes tf parameters
def initialize_parameters(n):

    # initializes W1 randomly as a 12 x 12 matrix
    W1 = tf.compat.v1.get_variable("W1", [12,12*n], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # initializes b as zeros as a 12 x 1 vector
    b1 = tf.compat.v1.get_variable("b1", [12,1], initializer = tf.zeros_initializer())
    # initializes W1 randomly as a 12 x 12 matrix
    W2 = tf.compat.v1.get_variable("W2", [12,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # initializes b as zeros as a 12 x 1 vector
    b2 = tf.compat.v1.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    # initializes W1 randomly as a 12 x 12 matrix
    W3 = tf.compat.v1.get_variable("W3", [12,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # initializes b as zeros as a 12 x 1 vector
    b3 = tf.compat.v1.get_variable("b3", [12,1], initializer = tf.zeros_initializer())
    # initializes W1 randomly as a 12 x 12 matrix
    W4 = tf.compat.v1.get_variable("W4", [12,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # initializes b as zeros as a 12 x 1 vector
    b4 = tf.compat.v1.get_variable("b4", [12,1], initializer = tf.zeros_initializer())
    # initializes W1 randomly as a 12 x 12 matrix
    W5 = tf.compat.v1.get_variable("W5", [12*n,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    # initializes b as zeros as a 12 x 1 vector
    b5 = tf.compat.v1.get_variable("b5", [12*n,1], initializer = tf.zeros_initializer())

    # returns parameters in dictionary
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5}

def forward_prop(X, parameters):
    # takes in input X and model parameters for Linear -> ReLu -> Linear -> ReLu -> Linear
    # consistent with the initialize params shapes
    # outputs output of the model
    
    Z1 = tf.add(tf.matmul(parameters['W1'],X),parameters['b1'])     # vectorized multiplication/addition step
    A1 = tf.nn.tanh(Z1)                                             # Tanh step    
    Z2 = tf.add(tf.matmul(parameters['W2'],A1),parameters['b2'])    # vectorized multiplication/addition step   
    A2 = tf.nn.tanh(Z2)                                          # Tanh step
    Z3 = tf.add(tf.matmul(parameters['W3'],A2),parameters['b3'])    # vectorized multiplication/addition step
    A3 = tf.nn.tanh(Z3)                                             # Tanh step    
    Z4 = tf.add(tf.matmul(parameters['W4'],A3),parameters['b4'])    # vectorized multiplication/addition step   
    A4 = tf.nn.tanh(Z4)                                             # Tanh step
    Z5 = tf.add(tf.matmul(parameters['W5'],A4),parameters['b5'])    # vectorized multiplication/addition step
    
    return tf.nn.sigmoid(Z5) # returns sigmoid of output

# cost function 
def compute_cost(Yhat, Y):
    # takes in true output Y and estimated output Yhat
    # returns cost as potision mean squared error
    return tf.reduce_mean(tf.keras.losses.MSE(Yhat[0:6], Y[0:6]))

def optimize_params(X_train, Y_train, X_dev, Y_dev, X_test, Y_test, n, learning_rate, num_epochs, minibatch_size):
    # takes in training X and Y, test X and Y, learning rate, number of epochs, and minibatch size
    # outputs optimized parameters
    
    # cost record
    costs = []
    train_costs = []
    dev_costs = []
    
    # placeholders for X and Y
    X = tf.compat.v1.placeholder(tf.float32, shape = [X_train.shape[0], None])
    Y = tf.compat.v1.placeholder(tf.float32, shape = [Y_train.shape[0] , None])

    # initial parameters
    parameters = initialize_parameters(n)
    
    # forward propagation
    Z5 = forward_prop(X, parameters)
    
    # first cost
    cost = compute_cost(Z5, Y)
    
    # implements Adam Optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # variable initialization
    init = tf.global_variables_initializer()

    # start session
    with tf.compat.v1.Session() as sess:
        
        # run initialization
        sess.run(init)
        
        for epoch in range(num_epochs): # loops over epochs

            epoch_cost = 0 # initialize zero epoch cost
            print(epoch)
            
            # number of minibatches
            num_minibatches = int(X_train.shape[1] / minibatch_size) 
            # defines minibatch
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:# loops over minibatches

                # runs optimizer on minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})                
                epoch_cost += minibatch_cost / minibatch_size
            
            # appends cost list
            costs.append(epoch_cost)
        
        # Calculates the estimated output from the test input
        Xtf_test = tf.convert_to_tensor(X_test)
        Xtf_test = tf.dtypes.cast(Xtf_test, dtype=tf.float32)      
        Y_hat_test = forward_prop(Xtf_test, parameters).eval(session=sess)
        
        # Transforms test set and estimated set into scalar values
        Y_test = binary_to_cartes(Y_test, n)
        Y_hat_test = binary_to_cartes(Y_hat_test, n)
        
        # Prints test MSE and standard deviation        
        print('Test Set Error Mean: ',np.square(Y_hat_test[0:6,:]-Y_test[0:6,:]).mean(axis=0).mean(axis=0))
        print('Test Set Error Standard Deviation: ',np.square(Y_hat_test[0:6,:]-Y_test[0:6,:]).mean(axis=0).std())

        # saves parameters returns final optimized parameters
        return sess.run(parameters)