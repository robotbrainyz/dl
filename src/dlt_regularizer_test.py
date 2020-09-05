import math
import torch
from dlt_regularizer import L2Regularizer

def test_L2_regularizer():
    regularizationFactor = 0.2
    regularizer = L2Regularizer(regularizationFactor)

    dummyWeight0 = torch.randn(3,2) # 2 input nodes, 3 nodes in middle hidden layer
    dummyWeight1 = torch.randn(1,3) # 1 output node, 3 nodes in middle hidden layer
    dummyWeights = [dummyWeight0, dummyWeight1]

    numExamples = 300

    regCost, regWeightsDelta = regularizer.regularize(dummyWeights, numExamples)

    expectedRegCost = torch.sum(torch.square(dummyWeight0)) + torch.sum(torch.square(dummyWeight1))
    expectedRegCost = (expectedRegCost * regularizationFactor)/(2 * numExamples)
    assert math.isclose(expectedRegCost, regCost, rel_tol=1e-05)

    expectedWeight0 = dummyWeight0 * regularizationFactor / numExamples
    expectedWeight1 = dummyWeight1 * regularizationFactor / numExamples

    assert torch.allclose(expectedWeight0, regWeightsDelta[0])
    assert torch.allclose(expectedWeight1, regWeightsDelta[1])    

    

    
    
    

