# API Documentation last updated 24 Oct 2020.

import torch

from dlt_activate import sigmoid_back, tanh_back

def back_linear(dz, a_prev, w):
    ''' Given dL/dz, compute dL/dw, dL/db, and dL/da_prev (dL/da_prev = dL/dx).

    Given that z=wx+b, a=activation(z), and L=loss(a) is the loss given a, this function computes dL/dw, dL/db, and dL/da_prev (dL/da_prev = dL/dx, and dL/da_prev is the dL/da for the previous layer).

    This function is used during back propagation.

    Args:
        dz (matrix): A (n[l] x m) matrix, where l is the current layer, n[l] is the number of nodes in the current layer, and m is the number of examples.

        a_prev (matrix): A (n[l-1] x m) matrix. This is the activation output from the previous layer during forward propagation.

        w (matrix): A (n[l] x n[l-1]) matrix. Weight values for the connections between l-1, the previous layer, and l, the current layer.

    Returns:
        dw (matrix): A (n[l] x n[l-1]) matrix. The values are the change in L with respect to the weights between l-1, the previous layer and l, the current layer.

        db (vector): A vector of size n[l]. The values are the change in L with respect to the biases b in l, the current layer.

        da_prev (matrix): A (n[l-1] x m) matrix. This is a matrix containing the change in L with respect to a[l-1], the activation output from l-1, the previous layer.

    '''
    
    m_inv = 1/dz.shape[1]
    dw = m_inv * torch.matmul(dz.float(), torch.transpose(a_prev, 0, 1).float())
    db = m_inv * torch.sum(dz, dim=1)
    db = db.reshape((len(db), 1))
    da_prev = torch.matmul(torch.transpose(w, 0, 1), dz.float())
    return dw, db, da_prev

def back_sigmoid(da, z):
    ''' Given dL/da, compute dL/dz. 

    Given that a = sigmoid(z), z = wx+b, and L=loss(a) is the loss given a, dL/dz = dL/da * sigmoid'(z).

    Args:
        da (matrix): A (n[l] x m) matrix containing dL/da, the change in loss L with respect to a, the activation output in l, the current layer.

        z (matrix): A (n[l] x m) matrix containing z=wx+b values in l, the current layer.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix representing the change in loss L with respect to z.
    '''    
    return da * sigmoid_back(z)

def back_tanh(da, z):
    ''' Given dL/da, compute dL/dz. 

    Given a = tanh(z), and z = wx+b, and L=loss(a) is the loss given a, dL/dz = dL/da * tanh'(z).

    Args:
        da (matrix): A (n[l] x m) matrix containing dL/da, the change in loss L with respect to a, the activation output in l, the current layer.

        z (matrix): A (n[l] x m) matrix containing z=wx+b values in l, the current layer.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix representing the change in loss L with respect to z.
    '''    
    return da * tanh_back(z)

def back_softmax(y, y_pred):
    ''' Given y and y_pred, compute dL/dz. 

    y_pred = a, where a is the softmax activation of z, and z=wx+b. See dlt_forward.forward for details on z=wx+b.

    Let L be the loss function given y and y_pred. By reduction of the underlying math, it is not necessary to compute dL/da to get dL/dz for back propagation. dL/dz is simply y_pred - y.

    Args:
        y (matrix): A (n[l] x m) matrix. Contains the expected output value for each feature in each example. Each example is a column in this matrix. m is the number of examples.

        y_pred (matrix): A (n[l] x m) matrix. Contains the predicted values for each example in each column. m is the number of examples.

    Returns:
        dL/dz (matrix): A (n[l] x m) matrix.    
    '''        
    return y_pred - y
    
def back(da, z, activationFunctionID):
    if activationFunctionID == 'sigmoid':
        return back_sigmoid(da, z)
    elif activationFunctionID == 'tanh':
        return back_tanh(da, z)
    else:
        assert(False) # Unrecognized activation function ID string    
    
