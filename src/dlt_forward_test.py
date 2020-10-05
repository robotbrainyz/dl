import math
import torch

from dlt_activate import sigmoid, tanh, softmax
from dlt_forward import forward, forward_rnn

def test_forward_sigmoid():
    # x is a 4x3 matrix. 3 examples, 4 features or nodes per example.
    xr = torch.randn(4, 3)

    # w is 2x4 matrix. 2 nodes in the layer being forward propagated. 4 nodes in the previous layer.
    wr = torch.randn(2, 4)

    # b is a 2x1 matrix, 2 constants for the 2 nodes in the layer being forward propagated.
    br = torch.randn(2, 1)

    # a, the forward propagation result in the layer is a 2x3 matrix. 2 nodes or features for each of the 3 examples.
    activationFunctionID = 'sigmoid'
    zr, ar = forward(xr, wr, br, activationFunctionID)

    # check the values of ar against manual matrix multiplication.
    wr0 = wr[0,:]
    xr0 = xr[:,0]
    wr1 = wr[1,:]
    xr1 = xr[:,1]
    xr2 = xr[:,2]        

    expected_value00 = 1.0/(math.exp(-torch.dot(wr0, xr0) - br[0][0]) + 1.0)
    expected_value10 = 1.0/(math.exp(-torch.dot(wr1, xr0) - br[1][0]) + 1.0)    

    expected_value01 = 1.0/(math.exp(-torch.dot(wr0, xr1) - br[0][0]) + 1.0)    
    expected_value11 = 1.0/(math.exp(-torch.dot(wr1, xr1) - br[1][0]) + 1.0)    

    expected_value02 = 1.0/(math.exp(-torch.dot(wr0, xr2) - br[0][0]) + 1.0)
    expected_value12 = 1.0/(math.exp(-torch.dot(wr1, xr2) - br[1][0]) + 1.0)
    
    assert math.isclose(ar[0][0], expected_value00, rel_tol=1e-05)
    assert math.isclose(ar[1][0], expected_value10, rel_tol=1e-05)

    assert math.isclose(ar[0][1], expected_value01, rel_tol=1e-05)
    assert math.isclose(ar[1][1], expected_value11, rel_tol=1e-05)
    
    assert math.isclose(ar[0][2], expected_value02, rel_tol=1e-05)
    assert math.isclose(ar[1][2], expected_value12, rel_tol=1e-05)

def test_forward_tanh():
    # x is a 4x3 matrix. 3 examples, 4 features or nodes per example.
    xr = torch.randn(4, 3)

    # w is 2x4 matrix. 2 nodes in the layer being forward propagated. 4 nodes in the previous layer.
    wr = torch.randn(2, 4)

    # b is a 2x1 matrix, 2 constants for the 2 nodes in the layer being forward propagated.
    br = torch.randn(2, 1)

    # a, the forward propagation result in the layer is a 2x3 matrix. 2 nodes or features for each of the 3 examples.
    activationFunctionID = 'tanh'
    zr, ar = forward(xr, wr, br, activationFunctionID)

    zExpected = torch.matmul(wr, xr) + br
    aExpected = tanh(zExpected)

    assert torch.allclose(zr, zExpected)
    assert torch.allclose(ar, aExpected)

def test_forward_softmax():
    # x is a 4x3 matrix. 3 examples, 4 features or nodes per example.
    xr = torch.randn(4, 3)

    # w is 2x4 matrix. 2 nodes in the layer being forward propagated. 4 nodes in the previous layer.
    wr = torch.randn(2, 4)

    # b is a 2x1 matrix, 2 constants for the 2 nodes in the layer being forward propagated.
    br = torch.randn(2, 1)

    # a, the forward propagation result in the layer is a 2x3 matrix. 2 nodes or features for each of the 3 examples.
    activationFunctionID = 'softmax'
    zr, ar = forward(xr, wr, br, activationFunctionID)

    zExpected = torch.matmul(wr, xr) + br
    aExpected = softmax(zExpected)

    assert torch.allclose(zr, zExpected)
    assert torch.allclose(ar, aExpected)        


def test_forward_rnn_sigmoid():
    xr = torch.randn(4, 3)
    aPrevr = torch.randn(2, 3)
    war = torch.randn(3, 6)
    bar = torch.randn(3, 1)

    activationFunctionID = 'sigmoid'
    z, a = forward_rnn(xr, aPrevr, war, bar, activationFunctionID)

    xa = torch.cat((aPrevr, xr), 0)
    assert xa.shape[0] == 6
    assert xa.shape[1] == 3

    zExpected = torch.matmul(war, xa) + bar
    assert torch.allclose(z, zExpected)
    
    aExpected = sigmoid(zExpected)
    assert torch.allclose(a, aExpected)

def test_forward_rnn_tanh():
    xr = torch.randn(4, 3)
    aPrevr = torch.randn(2, 3)
    war = torch.randn(3, 6)
    bar = torch.randn(3, 1)

    activationFunctionID = 'tanh'
    z, a = forward_rnn(xr, aPrevr, war, bar, activationFunctionID)

    xa = torch.cat((aPrevr, xr), 0)
    assert xa.shape[0] == 6
    assert xa.shape[1] == 3

    zExpected = torch.matmul(war, xa) + bar
    assert torch.allclose(z, zExpected)
    
    aExpected = tanh(zExpected)
    assert torch.allclose(a, aExpected)

def test_forward_rnn_softmax():
    xr = torch.randn(4, 3)
    aPrevr = torch.randn(2, 3)
    war = torch.randn(3, 6)
    bar = torch.randn(3, 1)

    activationFunctionID = 'softmax'
    z, a = forward_rnn(xr, aPrevr, war, bar, activationFunctionID)

    xa = torch.cat((aPrevr, xr), 0)
    assert xa.shape[0] == 6
    assert xa.shape[1] == 3

    zExpected = torch.matmul(war, xa) + bar
    assert torch.allclose(z, zExpected)
    
    aExpected = softmax(zExpected)
    assert torch.allclose(a, aExpected)        
    

