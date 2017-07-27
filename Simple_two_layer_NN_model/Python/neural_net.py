from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

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
    #self.params['b1'] = np.zeros(hidden_size)
    self.params['b1'] = 0.1 * np.array(range(hidden_size))
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    #self.params['b2'] = np.zeros(output_size)
    self.params['b2'] = 0.2 * np.array(range(output_size))

  def loss(self, X, y=None, reg=0.0):
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
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    #print('X=',X)
    
    max_before_h1 = X.dot(W1)+ b1
    #print('before_max_h1 =',max_before_h1)
    
    
    
    h1 = np.maximum(0, max_before_h1)
    #print('h1=',h1)
    
    h2 = h1.dot(W2) + b2
    #print('h2=',h2)
    
    scores = h2


    #############################################################################
    #                              END OF forwardpass                           #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # Finish the forward pass, and compute the loss. Store the result           #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    pass
   
    
    h2_exp = np.exp(h2)
    h2_expnorm_eachsample = h2_exp.sum(1)
    self.loss_eachsample = np.log(h2_expnorm_eachsample) - h2[range(N), y]
    #print('loss_eachsampe = ', loss_eachsample)
    loss = np.average(self.loss_eachsample)

    #############################################################################
    #                              END OF loss calculation                      #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # Compute the backward pass, computing the derivatives of the weights       #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    hidsize,outsize = h2.shape
    
    # gradient from regulazation
    grads['W1'] = reg*2*W1
    grads['W2'] = reg*2*W2
    grads['b1'] = np.zeros(b1.shape)
    grads['b2'] = np.zeros(b2.shape)
    
    
    
    # gradient from loss

    grad_1 = 1 / N  # grad of average
    grad_2 = grad_1*(-1/ (h2_exp[range(N),y]/h2_expnorm_eachsample)) # grad by logarithm, result in size [N 1]
    #print('grad_2.shape =', grad_2.shape)
    
    # grad of the expression: exp(h_(i,y_i)) / sigma (exp(h_ij)) ;
    # let exp(h_(i,y_i)) = a,  exp(h_(i,y_i))=b
    grad_tmp_1 = h2_exp[range(N),y] / h2_expnorm_eachsample 
    grad_tmp_1 = np.outer( grad_tmp_1,  np.ones(outsize) )
    #print('grad_tmp_1 =', grad_tmp_1)
    
    
    sel = np.zeros(h2.shape)
    sel[range(N),y] = 1
    grad_tmp_1 = grad_tmp_1 * sel  # get (a)' / b  , in size [N  outsize]
    
    grad_tmp_2 = - h2_exp[range(N),y]/ np.square(h2_expnorm_eachsample)
    grad_tmp_2 = np.outer( grad_tmp_2, np.ones(outsize) )
    grad_tmp_2 = grad_tmp_2 * h2_exp # get - a/b^2 * (b)'  in size [N outsize]
    
    grad_h2 = np.outer( grad_2, np.ones(outsize) ) * (grad_tmp_1 + grad_tmp_2) # grad of h2
    
    
    grad_b2 = grad_h2.sum(0) # grad_b2 = sum of each column of (grad_h2) 

    
    grad_h1 = grad_h2 .dot(W2.T) # grad_h1
    grad_W2 = h1.T .dot (grad_h2) # grad_W2
    
    #print('max_before_h1 = ', max_before_h1)
    #print('grad_h1 = ', grad_h1)
    grad_max_before_h1 = grad_h1
    grad_max_before_h1[max_before_h1<0] = 0
    
    grad_b1 = grad_max_before_h1.sum(0)
    grad_W1 = X.T .dot (grad_max_before_h1)
    grad_X  = grad_max_before_h1 .dot(W1.T)
    
    grads['W1']+= grad_W1
    grads['b1']+= grad_b1
    grads['W2']+= grad_W2
    grads['b2']+= grad_b2
    

    #############################################################################
    #                              END OF backwards                             #
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

    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # Create a random minibatch of training data and labels, storing        #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      N,D = X.shape
      label_batch = np.random.randint(0, high=N, size=batch_size)
      X_batch = X[label_batch]
      y_batch = y[label_batch]

      #########################################################################
      #                             END OF minibatch                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # Use the gradients in the grads dictionary to update the               #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      self.params['W1']-= grads['W1'] * learning_rate
      self.params['b1']-= grads['b1'] * learning_rate
      self.params['W2']-= grads['W2'] * learning_rate
      self.params['b2']-= grads['b2'] * learning_rate
    
      #########################################################################
      #                             END OF GD                                 #
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
    # Predict the classifier based on trained weights                         #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']    
    
    
    h1 = np.maximum(0, (X.dot(W1)+b1 ) )
    h2 = h1.dot(W2) + b2
    y_pred= np.argmax(h2, axis=1)  

    l1 = X.dot(W1) + np.resize(b1, (1, b1.shape[0]))
    scores = np.maximum(0, l1).dot(W2) + np.resize(b2, (1, b2.shape[0]))
    y_pred = np.argmax(scores, axis=1)
    
    
    ###########################################################################
    #                              END OF prediction                          #
    ###########################################################################

    return y_pred


