import math
import torch

from dlt_activate import sigmoid, tanh
from dlt_back import back_linear, back_sigmoid, back_tanh, back_softmax

def test_back_linear():
    dz = torch.randn(3,4) # 3 nodes in layer, 4 examples
    a_prev = torch.randn(2, 4) # 2 nodes in previous layer, 4 examples
    w = torch.randn(3, 2) # 3 nodes in layer, 2 nodes in previous layer

    dw, db, da_prev = back_linear(dz, a_prev, w)

    m_inv = 1/dz.shape[1]
    a_prev_T = torch.transpose(a_prev, 0, 1)
    w_T = torch.transpose(w, 0, 1)

    # Check value at row index 1, column index 1 of dw
    expected_value_dw = m_inv * torch.sum(dz[1, :] * a_prev_T[:, 1])
    math.isclose(dw[1, 1], expected_value_dw)

    # Check value at row index 2 of db
    math.isclose(db[2], m_inv * torch.sum(dz[2, :]))

    # Check value at row index 0, column index 1 of da_prev
    expected_value_da_prev = torch.sum(w_T[0, :] * dz[:, 1])
    math.isclose(da_prev[0, 1], expected_value_da_prev)
    
def test_back_sigmoid():
    da = torch.randn(3,4) # 3 nodes in layer, 4 examples
    z = torch.randn(3,4) # 3 nodes in layer, 4 examples

    dz = back_sigmoid(da, z)

    a = sigmoid(z)
    expected_value = da[1, 2] * (a * (1-a))[1, 2]
    math.isclose(dz[1, 2], expected_value)

def test_back_tanh():
    da = torch.randn(3,4) # 3 nodes in layer, 4 examples
    z = torch.randn(3,4) # 3 nodes in layer, 4 examples

    dz = back_tanh(da, z)

    a = tanh(z)
    expected_value = da[1, 2] * (1 - a * a)[1, 2]
    math.isclose(dz[1, 2], expected_value)

def test_back_softmax():
    y = torch.randn(3,4) # 3 nodes in layer, 4 examples
    y_pred = torch.randn(3,4) # 3 nodes in layer, 4 examples
    dz = back_softmax(y, y_pred)
    expected_value = y_pred - y

    # test that all values are equal to the expected values
    torch.allclose(dz, expected_value) 

