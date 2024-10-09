from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3



class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """



    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)



    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two-layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2'] #shapes 10,3 -- 3
        N, D = X.shape

        # Compute the forward pass
        scores = np.zeros((N, 3)) # (5,3)

        
        #############################################################################
        # TODO: Perform the forward pass, computing the class probabilities for the #
        # input. Store the result in the scores variable, which should be an array  #
        # of shape (N, C).                                                          #
        #############################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        a1 = X # a1 (5x4), W1 (4x10), b1 (10,)
        z2 = W1.T.dot(a1.T) + b1.reshape(-1, 1)  #z2 (10x5)
        a2 = np.maximum(0, z2)  # ReLU activation # a2 (10x5)
        z3 = a2.T.dot(W2) + b2 # z3 (5x3) W2 (10x3) b2 (3,)
        z3 -= np.max(z3, axis=1, keepdims=True) # for numerical stability
        scores = np.exp(z3) / np.sum(np.exp(z3), axis=1, keepdims=True) #(5,3) # Softmax activation
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


        # If the targets are not given then jump out, we're done
        if y is None:
            return scores


        # Compute the loss
        loss = 0. #scalar
        
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss.                                                          #
        #############################################################################
        
        # Implement the loss for the softmax output layer
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Cross-entropy loss
        Q = scores[range(N), y] # Q (5,)  scores[i, y[i]] 
        loss = 1/N * np.sum(-np.log(Q)) 
        # loss + L2 regularization
        loss += reg * (np.sum(W1**2)+np.sum(W2**2))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Backward pass: compute gradients
        grads = {}

        ##############################################################################
        # TODO: Implement the backward pass, computing the derivatives of the weights#
        # and biases. Store the results in the grads dictionary. For example,        #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size  #
        ##############################################################################

        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        delta = np.zeros_like(scores)
        delta[range(N), y] = 1
        dL_dz3 = 1/N * (scores - delta) # (5,3)
        """
        print(z3.shape) #(5,3)
        print (a2.shape) #(10,5)
        print (W2.shape) #(10,3)
        print(b2.shape) #(3,)
        print (a1.shape) #(5,4)
        print (W1.shape) #(4,10)
        print(b1.shape) #(10,)
        print((reg*W2).shape) #(10,3)
        """
        grads['W2'] = a2.dot(dL_dz3) + 2*reg*W2 # (10,5) * (5,3) + (10,3) = (10,3)
        grads['b2'] = np.sum(dL_dz3, axis=0) # (3,)
        dL_dz2 = dL_dz3.dot(W2.T) * ((z2.T > 0)) # (5,3) * (3,10) * (10,5) = (5,5)
        grads['W1'] = (a1.T.dot(dL_dz2)) + 2*reg*W1 # (4,5) * (5,10) + (4,10) = (4,10)
        grads['b1'] = np.sum(dL_dz2, axis=0) #(10)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return loss, grads



    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array of shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        
        num_train = X.shape[0]
        iterations_per_epoch = max( int(num_train // batch_size), 1)


        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = X
            y_batch = y

            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            
            # select batch_size random indices and use them to index into X and y
            idx_batch = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx_batch]
            y_batch = y[idx_batch]
            
        
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # At every epoch check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }



    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred


