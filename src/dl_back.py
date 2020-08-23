import numpy as np

from dl_activate import sigmoid_back, tanh_back

def back_linear(dz, a_prev, w):
    ''' Given dL/dz, compute dL/dw, dL/db, and dL/da_prev.
    '''
    
    m_inv = 1/dz.shape[1]
    dw = m_inv * np.matmul(dz, np.transpose(a_prev))
    db = m_inv * np.sum(dz, axis=1)
    da_prev = np.matmul(np.transpose(w), dz)
    return dw, db, da_prev

def back_sigmoid(da, z):
    ''' Given dL/da, compute dL/dz. 

    If a = sigmoid(z), dL/dz = dL/da * sigmoid'(z).
    '''    
    return da * sigmoid_back(z)

def back_tanh(da, z):
    ''' Given dL/da, compute dL/dz. 

    If a = tanh(z), dL/dz = dL/da * tanh'(z).
    '''    
    return da * tanh_back(z)

def back_softmax(y, y_pred):
    ''' Given y and y_pred (y_pred is also a), compute dL/dz. 
    '''        
    return y_pred - y
    
    

