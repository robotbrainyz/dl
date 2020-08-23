import math
import numpy as np

from dl_loss import loss_cross_entropy, loss_cross_entropy_back, loss_cross_entropy_softmax

def test_loss_cross_entropy_single_row_input():
    y = np.array([[1, 0, 1, 0]]) # (1 x 4) matrix
    y_pred = np.array([[0.9, 0.3, 0.1, 0.77]]) # (1 x 4) matrix
    L = loss_cross_entropy(y, y_pred) # (1 x 4) matrix 

    assert L[0][0] == -math.log(y_pred[0][0])
    assert L[0][1] == -math.log(1.0-y_pred[0][1])
    assert L[0][2] == -math.log(y_pred[0][2])
    assert L[0][3] == -math.log(1.0-y_pred[0][3])

def test_loss_cross_entropy_multi_row_input():
    y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) # (2 x 4) matrix
    y_pred = np.array([[0.9, 0.3, 0.1, 0.77], [0.83, 0.27, 0.119, 0.71]]) # (2 x 4) matrix
    L = loss_cross_entropy(y, y_pred) # 1D array with 4 elements

    assert L[0][0] == -math.log(y_pred[0][0])
    assert L[1][0] == -math.log(1.0-y_pred[1][0])
    assert L[0][1] == -math.log(1.0-y_pred[0][1])
    assert L[1][1] == -math.log(y_pred[1][1])
    assert L[0][2] == -math.log(y_pred[0][2])
    assert L[1][2] == -math.log(1.0 - y_pred[1][2])
    assert L[0][3] == -math.log(1.0-y_pred[0][3])
    assert L[1][3] == -math.log(y_pred[1][3])

def test_loss_cross_entropy_back():
    y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) # (2 x 4) matrix
    y_pred = np.array([[0.9, 0.3, 0.1, 0.77], [0.83, 0.27, 0.119, 0.71]]) # (2 x 4) matrix
    da = loss_cross_entropy_back(y, y_pred) # (2 x 4) matrix

    assert da[0][0] == -(y[0][0]/y_pred[0][0])
    assert da[0][1] == (1-y[0][1])/(1-y_pred[0][1])
    assert da[0][2] == -(y[0][2]/y_pred[0][2])
    assert da[0][3] == (1-y[0][3])/(1-y_pred[0][3])

    assert da[1][0] == (1-y[1][0])/(1-y_pred[1][0])
    assert da[1][1] == -(y[1][1]/y_pred[1][1])
    assert da[1][2] == (1-y[1][2])/(1-y_pred[1][2])
    assert da[1][3] ==  -(y[1][3]/y_pred[1][3])
    
def test_loss_cross_entropy_softmax():
    y = np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) # (2 x 4) matrix
    y_pred = np.array([[0.9, 0.3, 0.1, 0.77], [0.83, 0.27, 0.119, 0.71]]) # (2 x 4) matrix    
    L = loss_cross_entropy_softmax(y, y_pred)

    # Check that each element is -(y * log(y_pred))
    for i in range(0, y.shape[0]):
        for j in range(0, y.shape[1]):
            assert L[i][j] == -(y[i][j] * math.log(y_pred[i][j]))
