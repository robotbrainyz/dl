import torch

def sigmoid(z):
    ''' Sigmoid activation.

    Applies the sigmoid function to the given matrix element-wise. Sigmoid is the function 1 / (1 + e**-z)), where ** is the operator in python representing 'to the power of'.

    Args:
        z (matrix): a (n[1] x m) matrix, where n[1] is the number of nodes in the layer, and m is the number of examples. z = wx + b, where w is a (n[1] x n[0]) weight matrix, and x is a (n[0] x m) input matrix for the layer. n[0] is the number of nodes in the previous layer before the layer being activated. b is a (n[1] x m) biases matrix where every column is the same.

    Returns:
        matrix: A n[1] x m matrix where every element is the sigmoid of the corresponding element in z. Sigmoid is the function 1 / (1 + e**-z).
    '''
    return 1.0 / (torch.exp(-z) + 1.0)

def tanh(z):
    ''' Tanh activation.

    Applies the tanh function to the given matrix element-wise. Tanh is the function (2 / (1 + e**-2z)) - 1, where ** is the operator in python representing 'to the power of'.

    Args:
        z (matrix): a (n[1] x m) matrix, where n[1] is the number of nodes in the layer, and m is the number of examples. z = wx + b, where w is a (n[1] x n[0]) weight matrix, and x is a (n[0] x m) input matrix for the layer. n[0] is the number of nodes in the previous layer before the layer being activated. b is a (n[1] x m) biases matrix where every column is the same.

    Returns: 
        matrix: A n[1] x m matrix where every element is the tanh of the corresponding element in z. Tanh is the function (2 / (1 + e**-2z)) - 1.
    '''
    
    return (2.0 / (torch.exp(-2 * z) + 1.0)) - 1

def softmax(z):
    ''' Softmax activation.

    Args:
        z (matrix): a (n[1] x m) matrix, where n[1] is the number of nodes in the layer, and m is the number of examples. z = wx + b, where w is a (n[1] x n[0]) weight matrix, and x is a (n[0] x m) input matrix for the layer. n[0] is the number of nodes in the previous layer before the layer being activated. b is a (n[1] x m) biases matrix where every column is the same.

    Returns: 
        matrix: A n[1] x m matrix where the elements in each column sum up to 1, and each element in a column is a 0 to 1 probability that it is the predicted class among the n[l] classes.
    '''
    maxOfColumns, maxOfColumnsIdx = torch.max(z, dim=0)
    expz = torch.exp(z - maxOfColumns) # subtract max of each column for numerical stability.
    return expz / torch.sum(expz, dim=0)

def sigmoid_back(z):
    '''
    Derivative of the sigmoid function.
    This is used in back propagation.
    The sigmoid derivative is sigmoid(z).(1 - sigmoid(z)).

    Args: 
        z (matrix): a (n[1] x m) matrix, where n[1] is the number of nodes in the layer, and m is the number of examples. z = wx + b, where w is a (n[1] x n[0]) weight matrix, and x is a (n[0] x m) input matrix for the layer. n[0] is the number of nodes in the previous layer before the layer being activated. b is a (n[1] x m) biases matrix where every column is the same.

    Returns:
        matrix: sigmoid'(z), also da/dz, a n[1] x m matrix showing the change in z with respect to a, where a = sigmoid(z).
    '''
    
    sz = sigmoid(z)
    return sz * (1 - sz)

def tanh_back(z):
    '''
    Derivative of the tanh function.
    This is used in back propagation.
    The tanh derivative is 1 - tanh(z) * tanh(z).

    args: 
        z (matrix): a (n[1] x m) matrix, where n[1] is the number of nodes in the layer, and m is the number of examples. z = wx + b, where w is a (n[1] x n[0]) weight matrix, and x is a (n[0] x m) input matrix for the layer. n[0] is the number of nodes in the previous layer before the layer being activated. b is a (n[1] x m) biases matrix where every column is the same.

    Returns:
        matrix: tanh'(z), also da/dz, a n[1] x m matrix showing the change in z with respect to a, where a = tanh(z).
    '''
    
    tz = tanh(z)
    return 1 - (tz * tz)
