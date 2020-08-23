import numpy as np

from dl_activate import sigmoid_back, tanh_back

def back_linear(dz, a_prev, w):
    ''' Given dL/dz, compute dL/dw, dL/db, and dL/da_prev.

    Given z=wx+b, a=activation(z), and L is the loss function given a this function computes dL/dw, dL/db, and dL/da_prev (the dL/da for the previous layer).

    Args:
        dz (matrix): A (n[l] x m) matrix, where l is the current layer, and n[l] is the number of nodes in the current layer. 
        a_prev (matrix): A (n[l-1] x m) matrix. This is the activation output from the previous layer.
        w (matrix): A (n[l] x n[l-1]) matrix. Weight values for the connections between the l-1, the previous layer, and l, the current layer.

    Returns:
        dw (matrix): A (n[l] x n[l-1]) matrix. The values are the change in L with respect to the weights between the l-1 and l layers.
        db (vector): A vector of size n[l]. The values are the change in L with respect to the constant b in the l layer.
        da_prev (matrix): A (n[l-1] x m) matrix. This is a matrix containing the change in L with respect to a[l-1], the activation output from the previous layer.

    '''
    
    m_inv = 1/dz.shape[1]
    dw = m_inv * np.matmul(dz, np.transpose(a_prev))
    db = m_inv * np.sum(dz, axis=1)
    da_prev = np.matmul(np.transpose(w), dz)
    return dw, db, da_prev

def back_sigmoid(da, z):
    ''' Given dL/da, compute dL/dz. 

    Given a = sigmoid(z), and z = wx+b, and L is the loss function given a, dL/dz = dL/da * sigmoid'(z).

    Args:
        da (matrix): A (n[l] x m) matrix containing dL/da values.
        z (matrix): A (n[l] x m) matrix containing z=wx+b values.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix.
    '''    
    return da * sigmoid_back(z)

def back_tanh(da, z):
    ''' Given dL/da, compute dL/dz. 

    Given a = tanh(z), and z = wx+b, and L is the loss function given a, dL/dz = dL/da * tanh'(z).

    Args:
        da (matrix): A (n[l] x m) matrix containing dL/da values.
        z (matrix): A (n[l] x m) matrix containing z=wx+b values.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix.
    '''    
    return da * tanh_back(z)

def back_softmax(y, y_pred):
    ''' Given y and y_pred (y_pred is also a), compute dL/dz. 

    y_pred is a = softmax(z), where z=wx+b.
    Let L be the loss function given y and y_pred. By reduction of the underlying math, it is not necessary to compute dL/da to get dL/dz for back propagation. dL/dz is simply y_pred - y.

    Args:
        y (matrix): A (n[l] x m) matrix. Contains the expected output value for each example in each column. m is the number of examples.
        y (matrix): A (n[l] x m) matrix. Contains the predicted values for each example in each column. m is the number of examples.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix.    
    '''        
    return y_pred - y
    
    

