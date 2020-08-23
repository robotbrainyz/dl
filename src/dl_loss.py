import numpy as np


def loss_cross_entropy(y, y_pred):
    ''' Given the dataset's output and the predicted output, compute the cross entropy loss for each example.

    Args:
        y (matrix): The dataset's output. y is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y are expected to be 0s or 1s.

        y_pred (matrix): The predicted output. y_pred is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y_pred are expected to be in the range [0, 1]..

    Returns:
        matrix: L , a (n[l] x m) matrix. Each element is the loss value for a feature in a training example.
    '''    
    return - (y * np.log(y_pred) + (1.0-y) * np.log(1.0-y_pred))


def loss_cross_entropy_back(y, y_pred):
    ''' Derivative of the cross entropy loss function with respect to (w.r.t.) y_pred.
    
    This is used in backward propagation.

    Input:
        y (matrix): The dataset's output. y is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y are expected to be 0s or 1s.

        y_pred (matrix): The predicted output. y_pred is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y_pred are expected to be in the range [0, 1]..

    Returns:
        matrix: da, a (n[l] x m) matrix. Each element in da is the change in y_pred with respect to a change in the loss value.
    '''    
    return -np.divide(y, y_pred) + np.divide((1-y), (1-y_pred))

def loss_cross_entropy_softmax(y, y_pred):
    ''' Given the dataset's output and the predicted output with softmax, compute the cross entropy loss for each example.

    Unlike loss_cross_entropy, loss_cross_entropy_softmax does not take into account error on zero y-values. It is only concerned about the error on feature with y-value equals 1. E.g. is a column in y is [0 1 0 0], this function is only concerned about the loss on the 2nd element. Hence, the second part of the loss function in loss_cross_entropy, i.e. (1.0-y) * np.log(1.0-y_pred) is not present in loss_cross_entropy_softmax. It is assumed that each column in the y matrix is a one-hot encoded vector.

    Args:
        y (matrix): The dataset's output. y is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y are expected to be 0s or 1s.

        y_pred (matrix): The predicted output. y_pred is a (n[l] x m) matrix, where l is the number of layers in the model, n[l] is the number of nodes or features in the l-th layer, and m is the number of examples. The elements in y_pred are expected to be in the range [0, 1]..

    Returns:
        matrix: L , a (n[l] x m) matrix. Each element is the loss value for a feature in a training example.
    '''    
    return - (y * np.log(y_pred))

    
