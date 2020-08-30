import numpy as np
import numpy.testing as npt
from dl_regularizer import L2Regularizer

def test_L2_regularizer():
    regularizationFactor = 0.2
    regularizer = L2Regularizer(regularizationFactor)

    dummyWeight0 = np.random.randn(3,2) # 2 input nodes, 3 nodes in middle hidden layer
    dummyWeight1 = np.random.randn(1,3) # 1 output node, 3 nodes in middle hidden layer
    dummyWeights = [dummyWeight0, dummyWeight1]

    numExamples = 300

    regCost, regWeightsDelta = regularizer.regularize(dummyWeights, numExamples)

    expectedRegCost = np.sum(np.square(dummyWeight0)) + np.sum(np.square(dummyWeight1))
    expectedRegCost = (expectedRegCost * regularizationFactor)/(2 * numExamples)
    npt.assert_approx_equal(expectedRegCost, regCost)

    expectedWeight0 = dummyWeight0 * regularizationFactor / numExamples
    expectedWeight1 = dummyWeight1 * regularizationFactor / numExamples

    npt.assert_array_almost_equal_nulp(expectedWeight0, regWeightsDelta[0])
    npt.assert_array_almost_equal_nulp(expectedWeight1, regWeightsDelta[1])    

    

    
    
    

