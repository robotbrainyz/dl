import math
import numpy as np

from dl_activate import sigmoid, sigmoid_back, tanh, tanh_back, softmax

'''
Test sigmoid activating a matrix of zeros.
'''
def test_sigmoid_zeros():
    z0 = np.zeros((2, 2))
    a0 = sigmoid(z0)
    assert a0[0][0] == 0.5
    assert a0[0][1] == 0.5
    assert a0[1][0] == 0.5
    assert a0[1][1] == 0.5

'''
Test sigmoid activating a matrix of ones.
'''
def test_sigmoid_ones():
    z1 = np.ones((2, 2))
    a1 = sigmoid(z1)
    expected_value = 1.0 / (math.exp(-1.0) + 1.0)
    assert a1[0][0] == expected_value
    assert a1[0][1] == expected_value
    assert a1[1][0] == expected_value
    assert a1[1][1] == expected_value

'''
Test sigmoid activating a matrix of random values.
'''    
def test_sigmoid_random():
    zr = np.random.randn(3,4)
    zr_copy = np.copy(zr)
    ar = sigmoid(zr)
    zr_shape = zr.shape

    # Check that every element in ar, which was activated by the sigmoid function implementation, is equal to 1 / (1 + e**z), where z is the original value of the element before activation.
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            assert ar[i][j] == 1.0 / (math.exp(-zr_copy[i][j]) + 1.0)

'''
Test getting the sigmoid derivative of a matrix of random values.
'''        
def test_sigmoid_back():
    zr = np.random.randn(2, 3)
    da_dz = sigmoid_back(zr)

    sz = sigmoid(zr)
    sz = sz * (1-sz)

    for i in range(0,2):
        for j in range(0,3):
            assert da_dz[i][j] == sz[i][j]

'''
Test tanh activation of a matrix of random values.
'''                    
def test_tanh():
    zr = np.random.randn(3,4)
    zr_copy = np.copy(zr)
    ar = tanh(zr)
    zr_shape = zr.shape

    # Check that each element in ar is equal to (e**x - e**-x)/(e**x + e**-x)
    # tanh is implemented as (2 / (1 + e**-2z)) - 1 in dl_activate
    epsilon = 0.00000000001
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            ex = math.exp(zr_copy[i][j]) # e**x
            emx = math.exp(-zr_copy[i][j]) # e**-x
            expected_value = (ex - emx)/(ex + emx)
            assert abs(ar[i][j] - expected_value) < epsilon

'''
Test getting the tanh derivative of a matrix of random values.
'''                    
def test_tanh_back():
    zr = np.random.randn(3,4)
    zr_copy = np.copy(zr)
    zr_copy = tanh(zr_copy)
    ar = tanh_back(zr)
    zr_shape = zr.shape

    # Check that each element is 1 - tanh(z) * tanh(z),
    # where z is an element in the zr matrix.
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            expected_value = 1.0 - (zr_copy[i][j] * zr_copy[i][j])
            assert ar[i][j] == expected_value

'''
Test the softmax activation function
'''
def test_softmax():
    zr = np.random.randn(3,4)
    zr_copy = np.copy(zr)
    ar = softmax(zr)

    zr_max = np.max(zr, axis=0) # Get the max of each column
    zr_copy = zr_copy - zr_max # Subtract the max of each column from all rows
    zr_copy = np.exp(zr_copy) # Take the exponent of all values
    zr_copy_sum_cols = np.sum(zr_copy, axis=0) # The the sum of each column
    expected_value = zr_copy / zr_copy_sum_cols # Divide each row by the sum of each column
    np.testing.assert_equal(ar, expected_value) # Check that ar is the same as the expected value.
    

