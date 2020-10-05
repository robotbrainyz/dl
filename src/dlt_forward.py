import torch
from dlt_activate import sigmoid, tanh, softmax

def forward_activate(z, activationFunctionID):
    if activationFunctionID == 'sigmoid':
        return sigmoid(z)
    elif activationFunctionID == 'tanh':
        return tanh(z)
    elif activationFunctionID == 'softmax':
        return softmax(z)
    else:
        assert(False) # Unrecognized activation function ID string    

def forward(x, w, b, activationFunctionID):
    ''' Forward propagate x, the input to this layer, with w and b, the weights in this layer, to obtain z. Activate z with the activation function to produce a.

    a = activate (wx + b)

    Args: 
        x (matrix): A (n[0] x m) matrix, where n[0] is the number of nodes in the previous layer, and m is the number of examples.

        w (matrix): A (n[1] x n[0]) matrix of weights in this layer, where n[1] is the number of nodes in this layer, 

        b (matrix): A (n[1] x 1) matrix of constants in this layer, where n[1] is the number of nodes in this layer.

    Returns:
        z: A (n[1] x m) matrix equals to wx + b.

        matrix: A (n[1] x m) matrix where (every n[1] x 1) column is the activated forward propagation output for an example.
    '''    
    z = torch.matmul(w, x.float()) + b
    return z, forward_activate(z, activationFunctionID)


def forward_rnn(x, aPrev, wa, ba, activationFunctionID):
    ''' Forward propagate concatenated aPrev and x, with wa and ba, the weights in this cell, to obtain z. Activate z with the activation function to produce a.

    Args:
        x (matrix): A (nx x m) matrix, where nx is the number of input features, and m is the number of examples.

        aPrev (matrix): A (na x m) matrix, where na is the number of activated features from the previous cell, and m is the number of examples.

        wa (matrix): A (ni x (na+nx)) matrix, where ni is the number of nodes in this layer.

        ba (matrix): A (ni x 1) matrix of constants in this layer, where ni is the number of nodes in this layer.

    Returns:
        z: A (ni x m) matrix, equals to wa.(aPrev, x) + ba, where (aPrev,x) is the concatenated aPrev and x.

        matrix: A (ni x m) matrix, the result of running the activation function on z.
    '''
    z = torch.matmul(wa, torch.cat((aPrev, x), 0)) + ba
    return z, forward_activate(z, activationFunctionID)    
    
    
