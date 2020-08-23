import numpy as np

from dl_activate import sigmoid, tanh
from dl_back import back_linear, back_sigmoid, back_tanh

def test_back_linear():
    dz = np.random.randn(3,4) # 3 nodes in layer, 4 examples
    a_prev = np.random.randn(2, 4) # 2 nodes in previous layer, 4 examples
    w = np.random.randn(3, 2) # 3 nodes in layer, 2 nodes in previous layer

    dw, db, da_prev = back_linear(dz, a_prev, w)

    m_inv = 1/dz.shape[1]
    a_prev_T = np.transpose(a_prev)
    w_T = np.transpose(w)

    # Check value at row index 1, column index 1 of dw
    expected_value_dw = m_inv * np.sum(dz[1, :] * a_prev_T[:, 1])
    epsilon = 0.00000000001
    assert abs(dw[1, 1] - expected_value_dw) < epsilon

    # Check value at row index 2 of db
    assert db[2] == m_inv * np.sum(dz[2, :])

    # Check value at row index 0, column index 1 of da_prev
    expected_value_da_prev = np.sum(w_T[0, :] * dz[:, 1])
    assert abs(da_prev[0, 1] - expected_value_da_prev) < epsilon
    
def test_back_sigmoid():
    da = np.random.randn(3,4) # 3 nodes in layer, 4 examples
    z = np.random.randn(3,4) # 3 nodes in layer, 4 examples

    dz = back_sigmoid(da, z)

    a = sigmoid(z)
    expected_value = da[1, 2] * (a * (1-a))[1, 2]
    assert dz[1, 2] == expected_value

def test_back_tanh():
    da = np.random.randn(3,4) # 3 nodes in layer, 4 examples
    z = np.random.randn(3,4) # 3 nodes in layer, 4 examples

    dz = back_tanh(da, z)

    a = tanh(z)
    expected_value = da[1, 2] * (1 - a * a)[1, 2]
    assert dz[1, 2] == expected_value    

def test_back_softmax():
    assert 1==1
