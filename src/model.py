#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 2 in                                             #
# INF5860 - Machine Learning for Image analysis                                 #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2018.03.01                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # TODO: Task 1
    layer_info = conf['layer_dimensions']
    layer_length = len(layer_info)
    params = {}
    
    for l in range(1,layer_length):
        normal = np.sqrt(2/layer_info[l-1]) 
        params['W_' + str(l)] = np.random.randn(layer_info[l-1], layer_info[l]) * normal
        params['b_' + str(l)] = np.zeros([layer_info[l],1])
    
    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 a)
    A = []
    if activation_function == 'relu':
        relu = Z
        relu[relu < 0] = 0
        return relu
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 2 b)
    soft_max = []
    soft_max = np.exp(Z - np.max(Z)) / np.sum(np.exp(Z - np.max(Z)), axis=0) 
    
    return soft_max


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # TODO: Task 2 c)
    if is_training == True:
        layer_info = conf['layer_dimensions']
        act_select = conf['activation_function']
        layer_length = len(layer_info)
        input_data = X_batch
        features = {}        
        Y_proposed = None
        weight_key = []
        bias_key = []
        
        features['A_0'] = X_batch
        
        for i in range(1,layer_length):
            #weight_key.append('W_' + str(i))
            #bias_key.append('b_' + str(i))
            features['Z_' + str(i)] = np.dot( params['W_' + str(i)].transpose() ,  input_data) + params['b_' + str(i)] 
            features['A_' + str(i)] = activation(np.dot( params['W_' + str(i)].transpose() ,  input_data) + params['b_' + str(i)] , act_select)
            input_data = features['A_' + str(i)]
            
        Y_proposed = softmax(features['Z_' + str(layer_length-1)] )
        
        return Y_proposed, features
    else:
        layer_info = conf['layer_dimensions']
        act_select = conf['activation_function']
        layer_length = len(layer_info)
        input_data = X_batch
        features = {}        
        Y_proposed = None
        weight_key = []
        bias_key = []
        
        features['A_0'] = X_batch
        
        for i in range(1,layer_length):
            #weight_key.append('W_' + str(i))
            #bias_key.append('b_' + str(i))
            features['Z_' + str(i)] = np.dot( params['W_' + str(i)].transpose() ,  input_data) + params['b_' + str(i)] 
            features['A_' + str(i)] = activation(np.dot( params['W_' + str(i)].transpose() ,  input_data) + params['b_' + str(i)] , act_select)
            input_data = features['A_' + str(i)]
            
        Y_proposed = softmax(features['Z_' + str(layer_length-1)] )
        print("Finish Training")
        return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_reference):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # TODO: Task 3
    
    cost = 0
    num_class = Y_proposed.shape[0]
    num_data = Y_proposed.shape[1]
    num_correct = np.zeros((num_class, num_data))
    num_correct = 0
    
    cost = -np.sum(np.sum(Y_reference*np.log(Y_proposed))) / num_data
    
    
    for i in range(0,num_data):
        num_correct = num_correct + (np.argmax(Y_proposed[:,i]) == np.argmax(Y_reference[:,i]) )
    
    
    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    # TODO: Task 4 a)
    if activation_function == 'relu':
        relu_output = Z
        relu_output[relu_output >= 0] = 1
        relu_output[relu_output < 0] = 0
        
        return relu_output
    else:
        print("Error: Unimplemented derivative of activation function: {}", activation_function)
        return None


def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # TODO: Task 4 b)
    layer_info=conf["layer_dimensions"]
    act_f=conf["activation_function"]
    layer_length = len(layer_info)
    num_data=Y_proposed.shape[1]
    temp=np.ones((num_data,1))
    grad_params={}
    
    error = Y_proposed-Y_reference
    
    
    for i in range(layer_length-1, 0, -1):
        if i==layer_length-1 :
            error = error
        else:
            error = activation_derivative(features["Z_"+str(i)],act_f)*(np.dot(params["W_"+str(i+1)],(error) ))
            
        grad_params["grad_W_"+str(i)]=1/num_data*np.dot(features["A_"+str(i-1)],(error).transpose()) 
        grad_params["grad_b_"+str(i)]=1/num_data*np.dot((error),temp)
        
        
    #grad_params["grad_W_2"]=1/num_data*np.dot(features["A_1"],(error).transpose())    
    #grad_params["grad_b_2"]=1/num_data*np.dot((error),temp)
    
    #error_1=activation_derivative(features["Z_1"],act_f)*(np.dot(params["W_2"],(error)))
    
    #grad_params["grad_W_1"]=1/num_data*np.dot(features["A_0"],(error_1.transpose()))
    #grad_params["grad_b_1"]=1/num_data*np.dot(error_1,temp)
    
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        updated_params: Updated parameter dictionary.
    """
    # TODO: Task 5
    learn_rate  = conf['learning_rate']
    ori_key = list(params.keys())
    update_key = []
    updated_params = {}
    
    for i in range(0,len(params.keys()) ):
        update_key.append("grad_"+ori_key[i])
    
    for i in range(0,len(ori_key)):
        ind_ori = ori_key[i]
        ind_up  = update_key[i]
        updated_params[ind_ori] = params[ind_ori] - learn_rate * grad_params[ind_up]
    
    return updated_params
