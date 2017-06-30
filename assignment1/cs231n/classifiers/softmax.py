import numpy as np
from random import shuffle
#from past.builtins import xrange

def xrange(num):
    return list(range(num))

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
      #compute loss for this example
      scores = X[i].dot(W)


        # tricks to prevent numeric instability, by makeing scores less than zero and thus we will not be deviding large numbers
        # ensure scores_temp<= 0
      scores -= np.max(scores)
      exp_sum = np.sum(np.exp(scores))
      softmax = np.exp(scores[y[i]]) / exp_sum
      loss -= np.log(softmax)
        #compute gradient for this example, column by column
      scores_p =np.exp(scores)/exp_sum
      for j in range(num_classes):
          if j == y[i]:
              dscore = scores_p[j] - 1
          else:
              dscore = scores_p[j]
          dW[:,j] += dscore * X[i]


  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  
  loss = 0.0
  dW = np.zeros_like(W)
  num_Train = y.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # f_i = WX_i
  # L_i = -f_yi + log Sum_j(e^f_j)
  # OR: L_i = -log(e^f_yi/(Sum_j(e^f_j)))
  #print("W: ")
  #print(W.shape)
  f = np.exp(np.dot(W.T, X.T)) #(10, numTrain)
  #print("f: (10, numTrain)")
  #print(f.shape)
  fsum = f.sum(axis=0)
  #print("fsum: numTrain")
  #print(fsum.shape)
  m = np.identity(10)[y] #(numTrain, 10)
  #print ("m: (numTrain, 10)")
  #print(m.shape)
  fyi = f[y,np.arange(0, num_Train)]
  #print("fyi: (numTrain)")
  #print(fyi.shape)
  L = -np.log(fyi/fsum)
  #print(L.shape)
  loss = L.sum()/num_Train
  #'?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

  # dLi/dfk = p_k -1(if y_i = k) dfk/dw = X
  # dLi/dw = Sum_k (p_k -1(if y_i = k)).X_k
  p_k = f/fsum
  minusYi = np.identity(10)[y]
  p_k = p_k.T - minusYi
  #print(p_k.shape)
  dLdW = np.dot(p_k.T, X)/num_Train
  #print(dLdW.shape)
  dW = dLdW.T
  
  return loss, dW

