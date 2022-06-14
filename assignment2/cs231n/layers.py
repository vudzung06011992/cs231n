from builtins import range
from functools import lru_cache as cache
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = x.shape[0]
    x_t = x.reshape(N, -1)
    out = x_t.dot(w) + b.reshape(1, -1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_shape = x.shape 
    N = x.shape[0]
    x_t = x.reshape(N, -1)
    db = np.sum(dout, axis=0)
    dw = x_t.T @ dout
    dx = dout @ w.T
    dx = dx.reshape(x_shape)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    positions = np.where(x < 0)
    out = np.copy(x)
    out[positions] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.ones(x.shape)
    positions = np.where(x < 0)
    dx[positions] = 0
    dx = dout * dx

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n_samples = x.shape[0]
    score_mtrix = np.copy(x)
    max_score = score_mtrix.max()
    exp_scores = np.exp(score_mtrix - max_score)
    # exp_allclass = np.sum(np.exp(score_mtrix), axis=1, keepdims=True)
    P_matrix = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    P_trueclass = P_matrix[range(n_samples), y].reshape(-1,1)
    loss = - np.sum(np.log(P_trueclass)) / n_samples

    coef = - np.ones(P_matrix.shape) * P_trueclass
    minus_p_true = 1 - P_trueclass
    coef[range(n_samples), y] = minus_p_true.reshape(-1,) # for true class
    coef *= -1/P_trueclass # derivative of log 
    coef = np.multiply(coef, P_matrix)
    dx = coef
    dx /= n_samples

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

def _node_x_minus_miu_forward(x, normalization="batchnorm"):
    """Compute forward pass for the node x - miu

    Args:
      - x: A numpy array containing input data, of shape (N, D)

    Returns a tuple of:
      - out: output of shape (N, D)
      - node_cache: (x, miu)
    """
    if normalization == "batchnorm":
      miu = np.mean(x, axis=0, keepdims=True)
    if normalization == "layernorm":
      miu = np.mean(x, axis=1, keepdims=True)
    out = x - miu
    cache = (x, miu, normalization)
    return out, cache

def _node_x_minus_miu_backward(dout, cache):
    """Compute backward pass for the node x - miu

    Args:
      - dout: Upstream derivative, of shape (N, D)
      - node_cache: Tuple of:
        - x: Input data, of shape (N, D)
        - miu: A numpy array containing average of x on axis 0, of shape (D, ) (for batchnorm) or (N, ) (for layernorm)

    Returns a tuple of:
      - dx: Gradient with respect to x, of shape (N, D)
    """
    x, _, normalization = cache
    N, D = x.shape
    if normalization == "batchnorm":
      dx = dout - np.mean(dout, axis=0, keepdims=True) 
    if normalization == "layernorm":
      dx = dout - np.mean(dout, axis=1, keepdims=True)
    return dx

def _node_std_forward(x_minus_miu, normalization):
    N, D = x_minus_miu.shape
    if normalization == "batchnorm":
      std = np.sqrt(np.mean(x_minus_miu ** 2, axis=0, keepdims=True) + 1e-5)
    if normalization == "layernorm":
      std = np.sqrt(np.mean(x_minus_miu ** 2, axis=1, keepdims=True) + 1e-5)
    
    cache = (x_minus_miu, std, normalization)
    return std, cache

def _node_std_backward(dout, cache):
    x_minus_miu, std, normalization = cache
    N, D = x_minus_miu.shape

    if normalization == "batchnorm":
      dx_minus_miu = x_minus_miu / (N*std) * dout

    if normalization == "layernorm":
      dx_minus_miu = x_minus_miu / (D*std) * dout

    return dx_minus_miu

def _node_x_minus_miu_over_std_forward(x_minus_miu, std, normalization):
    
    x_minus_miu_over_std = x_minus_miu / std 
    cache = (x_minus_miu, std, normalization)
    
    return x_minus_miu_over_std, cache

def _node_x_minus_miu_over_std_backward(dout, cache):
    x_minus_miu, std, normalization = cache 

    dx_minus_miu = dout * 1/std
    if normalization == "batchnorm":
      dstd = (-1) * np.sum(dout * x_minus_miu, axis=0, keepdims=True) / (std)**2
    if normalization == "layernorm":
      dstd = (-1) * np.sum(dout * x_minus_miu, axis=1, keepdims=True) / (std)**2
    
    return dx_minus_miu, dstd

def _node_add_gamma_beta_forward(x, gamma, beta):
    return x * gamma + beta, (x, gamma, beta)
  
def _node_add_gamma_beta_backward(dout, cache):
    x, gamma, beta = cache
    dgamma = np.sum(dout * x, axis=0)
    dbeta = np.sum(dout, axis=0)
    dx = dout * gamma
    return dx, dgamma, dbeta

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - node_cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, node_cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        # Code with computation graph. Just use for reference
        x_minus_miu, x_minus_miu_cache = _node_x_minus_miu_forward(x, normalization="batchnorm")
        std, std_cache = _node_std_forward(x_minus_miu, normalization="batchnorm")
        node_x_minus_miu_over_std, x_minus_miu_over_std_cache = _node_x_minus_miu_over_std_forward(x_minus_miu, std, normalization="batchnorm")
        out, add_gamma_beta_cache = _node_add_gamma_beta_forward(node_x_minus_miu_over_std, gamma, beta)
        
        bn_param["mode"] = "train"
        
        node_cache = (x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache)
        _, miu, _ = x_minus_miu_cache
        var = std ** 2
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        n_train = x.shape[0]
        sample_mean = np.sum(x, axis=0)/n_train
        sample_var = np.var(x, axis=0)

        running_mean = bn_param["running_mean"]
        running_var = bn_param["running_var"]
        miu = momentum * running_mean + (1 - momentum) * sample_mean
        var = momentum * running_var + (1 - momentum) * sample_var
        out = (x - miu) / np.sqrt(var)
        out = out * gamma + beta
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = miu
    bn_param["running_var"] = var

    return out, node_cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - node_cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache = cache
    dx_minus_miu_over_std, dgamma, dbeta = _node_add_gamma_beta_backward(dout, add_gamma_beta_cache)
    dx_minus_miu_1, dstd = _node_x_minus_miu_over_std_backward(dx_minus_miu_over_std, x_minus_miu_over_std_cache)
    dx_minus_miu_2 = _node_std_backward(dstd, std_cache)
    dx_minus_miu = dx_minus_miu_1 + dx_minus_miu_2
    dx = _node_x_minus_miu_backward(dx_minus_miu, x_minus_miu_cache)
                            
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    miu_cache, x_minus_miu_cache, var_cache, std_cache, one_over_std_cache, batchnorm_comp_cache = cache

    std = one_over_std_cache 
    x, miu = x_minus_miu_cache
    var = std_cache
    gamma, beta, x_minus_miu, one_over_std = batchnorm_comp_cache
    N_train = x.shape[0]

    first_component = dout * one_over_std
    second_component = - np.sum(dout, axis=0) * 1/N_train * one_over_std
    third_component = - np.sum(x_minus_miu * dout, axis=0) * x_minus_miu * 1/N_train * (1/std)**3
    dx = first_component + second_component + third_component
    dx = dx * gamma
    x_t = x_minus_miu * one_over_std
    
    dgamma = np.sum(dout * x_t, axis=0)
    dbeta = np.sum(dout, axis=0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_minus_miu, x_minus_miu_cache = _node_x_minus_miu_forward(x, normalization="layernorm")
    std, std_cache = _node_std_forward(x_minus_miu, normalization="layernorm")
    node_x_minus_miu_over_std, x_minus_miu_over_std_cache = _node_x_minus_miu_over_std_forward(x_minus_miu, std, normalization="layernorm")
    out, add_gamma_beta_cache = _node_add_gamma_beta_forward(node_x_minus_miu_over_std, gamma, beta)
    cache = (x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache = cache
    dx_minus_miu_over_std, dgamma, dbeta = _node_add_gamma_beta_backward(dout, add_gamma_beta_cache)
    dx_minus_miu_1, dstd = _node_x_minus_miu_over_std_backward(dx_minus_miu_over_std, x_minus_miu_over_std_cache)
    dx_minus_miu_2 = _node_std_backward(dstd, std_cache)
    dx_minus_miu = dx_minus_miu_1 + dx_minus_miu_2
    dx = _node_x_minus_miu_backward(dx_minus_miu, x_minus_miu_cache)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask = (np.random.rand(*x.shape) < p)/p
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dropout_param, mask = cache
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    pad = conv_param["pad"]
    stride = conv_param["stride"]
    
    new_H = H + 2 * pad
    new_W = W + 2 * pad
    
    pad_x = np.zeros((N, C, new_H, new_W))
    # padding
    
    for i in range(C):
      pad_x[:, i, pad: new_H-pad, pad: new_W-pad] =  x[:, i, :, :]
    
    out_H = 1 + int((H + 2 * pad - HH) / stride)
    out_W = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, out_H, out_W))
    # loop over filter
    
    from itertools import product
    
    combs = product(range(F), range(out_H), range(out_W))
    for t, i, j in combs:
      filter_fc = w[t, :, :, :].flatten()
      position_h = i * stride
      position_w = j * stride
      pad_x_area = pad_x[:, :, position_h: position_h + HH, position_w: position_w + WW]
      pad_x_area_fc = pad_x_area.reshape(N, -1)
      
      out[:, t, i, j] = np.sum(pad_x_area_fc * filter_fc, axis=1) + b[t]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pad_x, w, b, conv_param)
    return out, cache

from itertools import product
def duplicate(testList, n):
  return [ele for ele in testList for _ in range(n)]

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pad_x, w, b, conv_param = cache
    pad = conv_param["pad"]
    stride = conv_param["stride"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    new_H = H + 2 * pad
    new_W = W + 2 * pad

    # tinh dx 
    out_H = 1 + int((H + 2 * pad - HH) / stride)
    out_W = 1 + int((W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, out_H, out_W))
    dpad_x = np.zeros(pad_x.shape)

    combs = product(range(N), range(F), range(out_H), range(out_W))

    # ban chat de tinh dpad_x thi di theo tung cell tren out. xac dinh area tren pad_x tuong ung. tinh dout * layer cua w tuong ung cho area trend pad_x do 
    for n, f, i, j in combs:
      position_h = i * stride
      position_w = j * stride
      dpad_x[n, :, position_h: position_h+HH, position_w: position_w+WW] += dout[n, f, i, j] * w[f, :, :, : ]
    
    dx = dpad_x[:, :, pad: new_H-pad, pad: new_W-pad]

    # tinh dw 
    dw = np.zeros(w.shape)
    combs = product(range(N), range(F), range(C), range(HH), range(WW))
    # de tinh dw thi kho hon mot chut. dau tien la voi tung wij tren w thi xac dinh cac cell tren pad_x ma nhan voi wij do. 
    # chinh la list_position_h (list cac index theo axis=2) va list_position_w (list cac index theo axis=3).
    for n, f, c, i, j in combs:
      list_position_h = [i + t for t in list(range(0, new_H-HH+1, stride))]
      list_position_w = [j + t for t in list(range(0, new_W-WW+1, stride))]
      len_h = len(list_position_h)
      len_w = len(list_position_w)
      
      # khi key hop list_position_h va list_position_w thi python se khong hieu de nhan decard. tu tao ra cac cap nhe.
      list_position_h = duplicate(list_position_h, len_w)
      list_position_w = list_position_w * len_h
      padx_given_layer_position = pad_x[n, c, list_position_h, list_position_w]
      
      # xac dinh cac score tuong ung tren out theo danh sach list_position_h va list_position_w
      list_position_h_out = [int(t/stride) for t in list(range(0, new_H-HH+1, stride))]
      list_position_w_out = [int(t/stride) for t in list(range(0, new_W-WW+1, stride))]
      len_h_out = len(list_position_h_out)
      len_w_out = len(list_position_w_out)
      # tuong tu, tu tao cac cap doi mot theo (h, w) thay vi python se hieu de nhan decard.
      list_position_h_out = duplicate(list_position_h_out, len_w_out)
      list_position_w_out = list_position_w_out * len_h_out
      
      dout_given_layer_position = dout[n, f, list_position_h_out, list_position_w_out]
      dw[f, c, i, j] += np.sum(padx_given_layer_position * dout_given_layer_position)
    
    # tinh db
    db = np.sum(dout, axis=(0, 2, 3))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    HH = int(1 + (H - pool_height) / stride)
    WW = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, HH, WW))
    combs = product(range(N), range(C), range(HH), range(WW))

    for n, c, i, j in combs:
      position_h_x = i * stride
      position_w_x = j * stride
      out[n, c, i, j] = np.max(x[n, c, position_h_x:position_h_x+pool_height, position_w_x:position_w_x+pool_width])

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    x, pool_param = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]
    HH = int(1 + (H - pool_height) / stride)
    WW = int(1 + (W - pool_width) / stride)

    out = np.zeros((N, C, HH, WW))
    combs = product(range(N), range(C), range(HH), range(WW))
    dx = np.zeros(x.shape)

    for n, c, i, j in combs:
      position_h_x = i * stride
      position_w_x = j * stride
      slide_x = x[n, c, position_h_x: position_h_x+pool_height, position_w_x: position_w_x+pool_width]
      maxi,maxj = np.unravel_index(slide_x.argmax(), slide_x.shape)
      dx[n, c, maxi + position_h_x, maxj + position_w_x] += dout[n, c, i, j]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, C, H, W = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(C, dtype=x.dtype))
    x_t = x.reshape(-1, C)
    out, node_cache = None, None
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_minus_miu, x_minus_miu_cache = _node_x_minus_miu_forward(x_t, normalization="batchnorm")
    std, std_cache = _node_std_forward(x_minus_miu, normalization="batchnorm")
    node_x_minus_miu_over_std, x_minus_miu_over_std_cache = _node_x_minus_miu_over_std_forward(x_minus_miu, std, normalization="batchnorm")
    out, add_gamma_beta_cache = _node_add_gamma_beta_forward(node_x_minus_miu_over_std, gamma, beta)
    out = out.reshape((N, C, H, W))    
    bn_param["mode"] = "train"
    
    cache = (x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = dout.shape
    dout_t = dout.reshape(-1, C)
    x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache = cache
    dx_minus_miu_over_std, dgamma, dbeta = _node_add_gamma_beta_backward(dout_t, add_gamma_beta_cache)
    dx_minus_miu_1, dstd = _node_x_minus_miu_over_std_backward(dx_minus_miu_over_std, x_minus_miu_over_std_cache)
    dx_minus_miu_2 = _node_std_backward(dstd, std_cache)
    dx_minus_miu = dx_minus_miu_1 + dx_minus_miu_2
    dx = _node_x_minus_miu_backward(dx_minus_miu, x_minus_miu_cache)
    dx = dx.reshape(N, C, H, W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    N, C, H, W = x.shape
    x_g = x.reshape((N*G,-1))
    
    x_minus_miu, x_minus_miu_cache = _node_x_minus_miu_forward(x_g, normalization="layernorm")
    std, std_cache = _node_std_forward(x_minus_miu, normalization="layernorm")
    node_x_minus_miu_over_std, x_minus_miu_over_std_cache = _node_x_minus_miu_over_std_forward(x_minus_miu, std, normalization="layernorm")
    node_x_minus_miu_over_std = node_x_minus_miu_over_std.reshape((N, C, H, W))
    out, add_gamma_beta_cache = _node_add_gamma_beta_forward(node_x_minus_miu_over_std, gamma, beta)
    cache = (x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache, G)
      
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x_minus_miu_cache, std_cache, x_minus_miu_over_std_cache, add_gamma_beta_cache, G = cache
    node_x_minus_miu_over_std, gamma, beta = add_gamma_beta_cache
    dx_minus_miu_over_std = dout * gamma
    N, C, H, W = dx_minus_miu_over_std.shape
    
    dgamma = np.sum(dout * node_x_minus_miu_over_std, axis=(0,2,3)).reshape(1, C, 1, 1)
    dbeta = np.sum(dout, axis=(0,2,3)).reshape(1, C, 1, 1)

    
    dx_minus_miu_over_std = dx_minus_miu_over_std.reshape((N*G,-1))

    dx_minus_miu_1, dstd = _node_x_minus_miu_over_std_backward(dx_minus_miu_over_std, x_minus_miu_over_std_cache)
    dx_minus_miu_2 = _node_std_backward(dstd, std_cache)
    dx_minus_miu = dx_minus_miu_1 + dx_minus_miu_2
    dx = _node_x_minus_miu_backward(dx_minus_miu, x_minus_miu_cache)
    dx = dx.reshape((N, C, H, W))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
