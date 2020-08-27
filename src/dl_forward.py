import numpy as np

from dl_activate import sigmoid, tanh, softmax

def forward_sigmoid(x, w, b):
    ''' Forward propagate x, the input to this layer, with w and b, the weights in this layer, to obtain z. Activate z with the sigmoid function to produce a.

    a = sigmoid (wx + b)

    Args: 
        x (matrix): A (n[0] x m) matrix, where n[0] is the number of nodes in the previous layer, and m is the number of examples.

        w (matrix): A (n[1] x n[0]) matrix of weights in this layer, where n[1] is the number of nodes in this layer, 

        b (matrix): A (n[1] x 1) matrix of constants in this layer, where n[1] is the number of nodes in this layer, 

    Returns:
        matrix: A (n[1] x m) matrix where (every n[1] x 1) column is the activated forward propagation output for an example.
    '''    
    return sigmoid(np.matmul(w, x) + b)

def forward_tanh(x, w, b):
    ''' Forward propagate x, the input to this layer, with w and b, the weights in this layer, to obtain z. Activate z with the tanh function to produce a.

    a = tanh (wx + b)

    Args: 
        x (matrix): A (n[0] x m) matrix, where n[0] is the number of nodes in the previous layer, and m is the number of examples.

        w (matrix): A (n[1] x n[0]) matrix of weights in this layer, where n[1] is the number of nodes in this layer, 

        b (matrix): A (n[1] x 1) matrix of constants in this layer, where n[1] is the number of nodes in this layer, 

    Returns:
        matrix: A (n[1] x m) matrix where (every n[1] x 1) column is the activated forward propagation output for an example.
    '''    
    return tanh(np.matmul(w, x) + b)

def forward_softmax(x, w, b):
    ''' Forward propagate x, the input to this layer, with w and b, the weights in this layer, to obtain z. Activate z with the softmax function to produce a.

    a = softmax (wx + b)

    Args: 
        x (matrix): A (n[0] x m) matrix, where n[0] is the number of nodes in the previous layer, and m is the number of examples.

        w (matrix): A (n[1] x n[0]) matrix of weights in this layer, where n[1] is the number of nodes in this layer, 

        b (matrix): A (n[1] x 1) matrix of constants in this layer, where n[1] is the number of nodes in this layer, 

    Returns:
        matrix: A (n[1] x m) matrix where (every n[1] x 1) column is the activated forward propagation output for an example.
    '''    
    return softmax(np.matmul(w, x) + b)

def forward(x, w, b, activationFunctionID):
    if activationFunctionID == 'sigmoid':
        return forward_sigmoid(x, w, b)
    elif activationFunctionID == 'tanh':
        return forward_tanh(x, w, b)
    elif activationFunctionID == 'softmax':
        return forward_softmax(x, w, b)
    else:
        assert(False) # Unrecognized activation function ID string
