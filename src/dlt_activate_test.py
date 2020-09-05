gimport math
import torch

from dlt_activate import sigmoid, sigmoid_back, tanh, tanh_back, softmax

'''
Test sigmoid activating a matrix of zeros.
'''
def test_sigmoid_zeros():
    z0 = torch.zeros((2, 2))
    a0 = sigmoid(z0)
    assert math.isclose(a0[0][0], 0.5, rel_tol=1e-05)
    assert math.isclose(a0[0][1], 0.5, rel_tol=1e-05)
    assert math.isclose(a0[1][0], 0.5, rel_tol=1e-05)
    assert math.isclose(a0[1][1], 0.5, rel_tol=1e-05)

'''
Test sigmoid activating a matrix of ones.
'''
def test_sigmoid_ones():
    z1 = torch.ones((2, 2))
    a1 = sigmoid(z1)
    expected_value = 1.0 / (math.exp(-1.0) + 1.0)
    assert math.isclose(a1[0][0], expected_value, rel_tol=1e-05)
    assert math.isclose(a1[0][1], expected_value, rel_tol=1e-05)
    assert math.isclose(a1[1][0], expected_value, rel_tol=1e-05)
    assert math.isclose(a1[1][1], expected_value, rel_tol=1e-05)

'''
Test sigmoid activating a matrix of random values.
'''    
def test_sigmoid_random():
    zr = torch.randn(3,4)
    zr_copy = zr.clone().detach()
    ar = sigmoid(zr)
    zr_shape = zr.shape

    # Check that every element in ar, which was activated by the sigmoid function implementation, is equal to 1 / (1 + e**z), where z is the original value of the element before activation.
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            assert math.isclose(ar[i][j], 1.0 / (math.exp(-zr_copy[i][j]) + 1.0), rel_tol=1e-05)

'''
Test getting the sigmoid derivative of a matrix of random values.
'''        
def test_sigmoid_back():
    zr = torch.randn(2, 3)
    da_dz = sigmoid_back(zr)

    sz = sigmoid(zr)
    sz = sz * (1-sz)

    for i in range(0,2):
        for j in range(0,3):
            assert math.isclose(da_dz[i][j], sz[i][j], rel_tol=1e-05)

'''
Test tanh activation of a matrix of random values.
'''                    
def test_tanh():
    zr = torch.randn(3,4)
    zr_copy = zr.clone().detach()
    ar = tanh(zr)
    zr_shape = zr.shape

    # Check that each element in ar is equal to (e**x - e**-x)/(e**x + e**-x)
    # tanh is implemented as (2 / (1 + e**-2z)) - 1 in dl_activate
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            ex = math.exp(zr_copy[i][j]) # e**x
            emx = math.exp(-zr_copy[i][j]) # e**-x
            expected_value = (ex - emx)/(ex + emx)
            assert math.isclose(ar[i][j], expected_value, rel_tol=1e-04)

'''
Test getting the tanh derivative of a matrix of random values.
'''                    
def test_tanh_back():
    zr = torch.randn(3,4)
    zr_copy = zr.clone().detach()
    zr_copy = tanh(zr_copy)
    ar = tanh_back(zr)
    zr_shape = zr.shape

    # Check that each element is 1 - tanh(z) * tanh(z),
    # where z is an element in the zr matrix.
    for i in range(0, zr_shape[0]):
        for j in range(0, zr_shape[1]):
            expected_value = 1.0 - (zr_copy[i][j] * zr_copy[i][j])
            assert math.isclose(ar[i][j], expected_value, rel_tol=1e-05)

'''
Test the softmax activation function
'''
def test_softmax():
    zr = torch.randn(3,4)
    zr_copy = zr.clone().detach()
    ar = softmax(zr)

    zr_max, zr_max_idx = torch.max(zr, dim=0) # Get the max of each column
    zr_copy = zr_copy - zr_max # Subtract the max of each column from all rows
    zr_copy = torch.exp(zr_copy) # Take the exponent of all values
    zr_copy_sum_cols = torch.sum(zr_copy, dim=0) # The the sum of each column
    expected_value = zr_copy / zr_copy_sum_cols # Divide each row by the sum of each column
    assert torch.allclose(ar, expected_value) # Check that ar is the same as the expected value.
    

