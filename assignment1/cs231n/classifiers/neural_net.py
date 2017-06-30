from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange
from cs231n.classifiers.softmax import softmax_loss_vectorized
from cs231n.classifiers.linear_svm import svm_loss_vectorized
import os

print(os.getcwd())
    
    
class TwoLayerNet(object):
  """
    N: input_size = 4
    H: hidden_size = 10
    C: num_classes = 3
    num_inputs = 5
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
    reg = reg/2 #For some reason, my regularization is off by half all the time.... 
    
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    Test: If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    Training: If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #print("b1:", b1.shape)
    p1 = np.dot(X,W1) + b1 # pass1
    '''
    OR: 
    reLU = lambda x: np.maximum(0, x)
    hidden_layer = reLU(X.dot(W1) + b1)
    '''
    L1 = (p1>0)*p1
    p2 = np.dot(L1,W2) + b2
    scores = p2
    
    '''
    margin1 = np.dot((p1>0), p1)
    #def svm_loss_vectorized(W, X, y, reg):
    l1, dW1 = svm_loss_vectorized(W1, X, y, reg)
    l2, dW2 = softmax_loss_vectorized(W1, P1, y, reg)
    '''
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    num_Train = y.shape[0]
    num_Classes = np.max(y) + 1
    num_Hidden = b1.shape[0]
    
    scores -= np.max(scores) # trick to avoid 0/0 situation
    f = np.exp(scores).T #(num_Classes, numTrain)
    fsum = f.sum(axis=0)
    fyi = f[y,np.arange(0, num_Train)]
    L = -np.log(fyi/fsum)
    loss = L.sum()/num_Train
    loss += 0.5 * reg * np.sum(W1**2)
    loss += 0.5 * reg * np.sum(W2**2)
    
    
    dP1dX = (p1>0)*p1
    '''
    prep1 = np.multiply.outer(np.identity(num_Hidden), X).T
    dP1dW1 = prep1
    dL1dW1 = np.reshape((p1>0), [1,5,10,1])*prep1
    dP1dB1 = np.identity(num_Hidden)
    dL1dB1 = np.dot((p1>0),dP1dB1)
    '''
    
    dP2dW2 = L1/num_Train
    dP2dB1 = 1
    dP2dL1 = W2
    
    p_k = f/fsum
    minusYi = np.identity(num_Classes)[y]
    dLidP2 = (p_k.T - minusYi).T
    dLdP2 = dLidP2.sum(axis=1)/num_Train

    #dLdW1 = Sum(dLdP2*dP2dp1*dP1dW1)
    #dLdW2 = Sum(dLdP2*dP2dW2)
    #dLdB1 = Sum(dLdP2*dP2dp1*dP1dB1)
    #dLdB2 = Sum(dLdP2*1)
    
    grads = {}
    #print("P1: ", p1.shape, ", W1: ", W1.shape)
    #print("dLdP2: ", dLdP2.shape,", dP2dL1: ",  dP2dL1.shape, ", dL1dW1: ", dL1dW1.shape, "dL1dB1: ", dL1dB1.shape, "P1: ", p1.shape, ", W1: ", W1.shape)
    #grads['W1'] = np.dot(dLdP2, dP2dL1.T).T
    #print(grads["W1"].shape)dLdbdLdb1dLdb1dLdb11
    #grads['W1'] = np.dot(grads['W1'], dL1dW1).sum(axis = 1)

    
    reLU = lambda x: np.maximum(0, x)
    hidden_layer = reLU(X.dot(W1) + b1)
    scores = hidden_layer.dot(W2) + b2
    
    scores -= np.max(scores)
    probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    dscores = probs
    dscores[np.arange(N), y] -= 1
    dscores /= N
    dhidden = dscores.dot(W2.T)
    dhidden[hidden_layer <= 0] = 0
    gw1 = X.T.dot(dhidden)
    gb1 = np.sum(dhidden, axis=0)

    
    dLidb1 = p_k.T.dot(W2.T)
    dLidb1 = ((p_k.T - minusYi)/num_Train).dot(W2.T)
    dLidb1[(p1<0)] = 0
    #dLidb1 = dLidb1*((p1<0)*1) # I don't know how to do this. :'(
    dLdb1 = dLidb1.sum(axis=0)
    grads['b1'] = dLdb1
    #print("blah: ", grads['b1'] )
    #print(gb1 )
    
    dLidW1 = dLidb1.T.dot(X)
    grads['W1'] = dLidW1.T

    grads['W2'] = np.dot(dLidP2, dP2dW2).T
    #print('W2: ', W2.shape, ', dLdW2: ', grads['W2'].shape)
    grads['b2'] = dLdP2*1
    #print('B2: ', b2.shape, ', dLdB2: ', grads['b2'].shape)

    grads['W1'] += reg * W1
    grads['W2'] += reg * W2

    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    #grads = {}
    #g1 = softmax_loss_vectorized()
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
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
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in np.arange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      mask = np.random.randint(num_train, size=batch_size)
      X_batch = X[mask]
      y_batch = y[mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for p in self.params: 
            self.params[p] -= grads[p]*learning_rate
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    
    reLU = lambda x: np.maximum(0, x)
    hidden_layer = reLU(X.dot(W1) + b1)
    scores = np.dot(hidden_layer,W2) + b2
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


