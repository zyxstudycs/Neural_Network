import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys

class NeuralNetwork(object):
    """
    Abstraction of neural network.
    Stores parameters, activations, cached values. 
    Provides necessary functions for training and prediction. 
    """
    def __init__(self, layer_dimensions, drop_prob=0.0, reg_lambda=0.0):
        """
        Initializes the weights and biases for each layer
        :param layer_dimensions: (list) number of nodes in each layer
        :param drop_prob: drop probability for dropout layers. Only required in part 2 of the assignment
        :param reg_lambda: regularization parameter. Only required in part 2 of the assignment
        """
        np.random.seed(1)
        
        self.parameters = {}
        self.num_layers =len(layer_dimensions)
        self.drop_prob = drop_prob
        self.reg_lambda = reg_lambda
        self.train_cost = []
        self.validation_cost = []
        self.train_accuracy = []
        self.validation_accuracy = []
        
        # init parameters
        self.parameters = self.init_parameters(layer_dimensions)
        
    def init_parameters(self, layer_dimensions):
        
        parameters = {}
        
        for i in range(len(layer_dimensions) - 1):
            forwoard_unit_number = layer_dimensions[i+1]
            backwoard_unit_number = layer_dimensions[i]
            parameters['W'+str(i+1)] = 0.01 * np.random.randn(forwoard_unit_number, backwoard_unit_number)
            parameters['b'+str(i+1)] = np.zeros((forwoard_unit_number,1))
            
        return parameters

    def affineForward(self, A, W, b):
        """
        Forward pass for the affine layer.
        :param A: input matrix, shape (L, S), where L is the number of hidden units in the previous layer and S is
        the number of samples
        :returns: the affine product WA + b, along with the cache required for the backward pass
        """
        return np.dot(W, A) + b

    def activationForward(self, A, activation="relu"):
        """
        Common interface to access all activation functions.
        :param A: input to the activation function
        :param prob: activation funciton to apply to A. Just "relu" for this assignment.
        :returns: activation(A)
        """ 
        return relu(A)

    def relu(self, X):
        return np.maximum(0,X)
            
    def dropout(self, A, prob):
        """
        :param A: 
        :param prob: drop prob
        :returns: tuple (A, M) 
            WHERE
            A is matrix after applying dropout
            M is dropout mask, used in the backward pass
        """
        M = np.random.rand(A.shape[0], A.shape[1])
        M = 1*(M >= prob)
        A = np.multiply(A, M)
        A = A / (1 - prob)
#         print('forward dropout is working**************')

        return A, M

    
    def forwardPropagation(self, X, dropout = True):
        cache = {}
        cache['A' + str(0)] = X
        
        for i in range(self.num_layers - 2):
            W = self.parameters['W' + str(i+1)]
            b = self.parameters['b' + str(i+1)]
            A = cache['A' + str(i)]
            Z = self.affineForward(A, W, b)
            A_next = self.relu(Z)
            
            # adding dropout here
            if self.drop_prob > 0 and dropout:
                A_next, M = self.dropout(A_next, self.drop_prob)
                cache['M' + str(i+1)] = M
                
            cache['A' + str(i+1)] = A_next
            cache['Z' + str(i+1)] = Z
        
        W = self.parameters['W' + str(self.num_layers - 1)]
        b = self.parameters['b' + str(self.num_layers - 1)]
        AL = self.affineForward(cache['A' + str(self.num_layers - 2)], W, b)

        return AL, cache
    
    def softmax(self, Z):
        K = Z - np.amax(Z, axis=0)
        K = np.exp(K)
        sum_ = np.sum(K,axis=0, keepdims=True)
        return K / sum_
    
    def costFunction(self, AL, y):
        # compute loss
        AL = self.softmax(AL)
        y_one_hot = np.zeros((AL.shape[0], AL.shape[1]))
        
        cost = 0
        cost = np.sum(-np.log(np.float64(AL[y, range(len(y))])))/len(y)
        y_one_hot[y, range(len(y))] = 1
        
        if self.reg_lambda > 0:
            for i in range(self.num_layers - 2):
                W = self.parameters['W' + str(i + 1)]
                cost += np.sum(np.multiply(W, W)) * self.reg_lambda / (2 * len(y))

        dAL = AL
        dAL[y, range(len(y))] -= 1
        
        return cost, dAL

    def affineBackward(self, dA_prev, cache, layer_num):
        """
        Backward pass for the affine layer.
        :param dA_prev: gradient from the next layer.
        :param cache: cache returned in affineForward
        :returns dA: gradient on the input to this layer
                 dW: gradient on the weights
                 db: gradient on the bias
        """
        m = dA_prev.shape[1]
        A = cache['A' + str(layer_num-1)]
        W = self.parameters['W' + str(layer_num)]
        
        #add dropout back propagation here
        if self.drop_prob > 0:
            dA_prev = self.dropout_backward(dA_prev, cache, layer_num)
        
        dZ = self.activationBackward(dA_prev, cache, layer_num)
        dW = 1 / m * np.dot(dZ, A.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA = np.dot(W.T, dZ)

        return dA, dW, db

    def activationBackward(self, dA, cache, layer_num, activation="relu"):
        """
        Interface to call backward on activation functions.
        In this case, it's just relu. 
        """
        dZ = dA * self.relu_derivative(cache['Z' + str(layer_num)])
        return dZ

        
    def relu_derivative(self, cached_x):
        dx = 1*(cached_x > 0)
        return dx

    def dropout_backward(self, dA, cache, layer_num):
        M = cache['M' + str(layer_num)]
        dA = np.multiply(M, dA) / (1 - self.drop_prob)
#         print('backward dropout is working *********')
        
        return dA

    def backPropagation(self, dAL, AL, cache):
        """
        Run backpropagation to compute gradients on all paramters in the model
        :param dAL: gradient on the last layer of the network. Returned by the cost function.
        :param Y: labels
        :param cache: cached values during forwardprop
        :returns gradients: dW and db for each weight/bias
        """
        gradients = {}
        
        m = AL.shape[1]
        A = cache['A' + str(self.num_layers - 2)]
        W = self.parameters['W' + str(self.num_layers - 1)]
        dZ = dAL
        dW = 1 / m * np.dot(dZ, A.T)
        db = 1 / m * np.sum(dZ, axis = 1, keepdims = True)
        dA = np.dot(W.T, dZ)
        
        gradients['dW' + str(self.num_layers - 1)] = dW
        gradients['db' + str(self.num_layers - 1)] = db
        
#         print('the dW' + str(self.num_layers - 1) + ' is' + str(np.sum(dW)/ (dW.shape[0] * dW.shape[1])))
#         print('the db' + str(self.num_layers - 1) + ' is' + str(np.sum(db)/ (db.shape[0] * db.shape[1])))
        
        for layer_num in range(self.num_layers - 2, 0, -1):
            dA, dW, db = self.affineBackward(dA, cache, layer_num)
            gradients['dW' + str(layer_num)] = dW
            gradients['db' + str(layer_num)] = db
            
#             print('the dW' + str(layer_num) + ' is' + str(np.sum(dW)/ (dW.shape[0] * dW.shape[1])))
#             print('the db' + str(layer_num) + ' is' + str(np.sum(db)/ (db.shape[0] * db.shape[1])))
        
        if self.reg_lambda > 0:
            # add gradients from L2 regularization to each dW
            for i in range(self.num_layers - 2):
                dW = gradients['dW' + str(i + 1)]
                W = self.parameters['W' + str(i + 1)]
                dW = dW + self.reg_lambda / m * W
                gradients['dW' + str(i + 1)] = dW
            
        return gradients


    def updateParameters(self, gradients, alpha):
        """
        :param gradients: gradients for each weight/bias
        :param alpha: step size for gradient descent 
        """
        for i in range(self.num_layers - 1):
            W = self.parameters['W' + str(i+1)]
            b = self.parameters['b' + str(i+1)]
            
            dW = gradients['dW' + str(i+1)]
            db = gradients['db' + str(i+1)]
            
#             print('the dW' + str(i+1) + ' is' + str(np.sum(dW) / (dW.shape[0] * dW.shape[1])))
#             print('the db' + str(i+1) + ' is' + str(np.sum(db)/ (db.shape[0] * db.shape[1])))


            W -= alpha * dW
            b -= alpha * db
            self.parameters['W' + str(i+1)] = W
            self.parameters['b' + str(i+1)] = b

    def train(self, X, y, X_validation, y_validation, iters=1000, alpha=0.1, batch_size=100, print_every=100):
        """
        :param X: input samples, each column is a sample
        :param y: labels for input samples, y.shape[0] must equal X.shape[1]
        :param iters: number of training iterations
        :param alpha: step size for gradient descent
        :param batch_size: number of samples in a minibatch
        :param print_every: no. of iterations to print debug info after
        """
        
        for i in range(0, iters):
            # get minibatch
            X_batch, y_batch = self.get_batch(X, y, batch_size)
            
            # forward prop
            AL, cache = self.forwardPropagation(X_batch)
            
            # compute loss
            cost, dAL = self.costFunction(AL, y_batch)
            
            # compute gradients
            gradients = self.backPropagation(dAL, AL, cache)
            
            # update weights and biases based on gradient
            self.updateParameters(gradients, alpha)
            
            if i % print_every == 0:
                # print cost, train and validation set accuracies
                train_y_predict = self.predict(X_batch)
                train_accuracy = np.sum((train_y_predict == y_batch) * 1) / len(y_batch)
                self.train_cost.append(cost)
                self.train_accuracy.append(train_accuracy)
                
                print('the training cost after %04d iteration is %8.6f:'%(i, cost))
                print('the training accuracy after %04d iteration is %8.2f:'%(i, train_accuracy))
                
                AL, _ = self.forwardPropagation(X_validation, dropout = False)
                validation_cost, _ = self.costFunction(AL, y_validation)
                validation_y_predict = self.predict(X_validation)
                validation_accuracy = np.sum((validation_y_predict == y_validation) * 1) / len(y_validation)
                self.validation_cost.append(validation_cost)
                self.validation_accuracy.append(validation_accuracy)
                print('the validation cost after %04d iteration is %8.6f:'%(i, validation_cost))
                print('the validation accuracy after %04d iteration is %8.2f:'%(i, validation_accuracy)) 
    
                            
    def predict(self, X):
        """
        Make predictions for each sample
        """
        AL,_ = self.forwardPropagation(X, dropout = False)
        y_pred = np.argmax(AL, axis=0)
        
        return y_pred

    def get_batch(self, X, y, batch_size):
        """
        Return minibatch of samples and labels
        
        :param X, y: samples and corresponding labels
        :parma batch_size: minibatch size
        :returns: (tuple) X_batch, y_batch
        """
        m = X.shape[1]
        start_index = np.random.randint(0, m - batch_size)
        X_batch = X[:, start_index:(start_index + batch_size)]
        y_batch = y[start_index:(start_index + batch_size)]

        return X_batch, y_batch