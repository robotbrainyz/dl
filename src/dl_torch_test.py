'''
np.random.shuffle
np.less
np.copy
np.vstack
np.arange
np.hstack
np.square

npt.assert_approx_equal
npt.assert_array_almost_equal_nulp

'''

import math
import torch

def test_torch_functions():
    '''
    np.ones
    np.zeros
    np.array
    np.random.randn
    '''
    # torch.ones - np.ones
    matOnes = torch.ones(2, 3)
    assert matOnes.shape[0] == 2
    assert matOnes.shape[1] == 3
    assert matOnes[0][0] == 1

    # torch.zeros - np.zeros
    matZeros = torch.zeros(2,3)
    assert matZeros.shape[0] == 2
    assert matZeros.shape[1] == 3
    assert matZeros[1][2] == 0

    # torch.tensor - np.array
    arr = torch.tensor([1, 2, 3])
    assert arr[1] == 2

    # torch.randn - np.random.randn
    matRandn = torch.randn(2, 3)
    
    '''
    np.exp
    np.max
    np.sum
    np.matmul
    np.transpose
    '''
    
    # torch.exp - np.exp
    matExp = torch.exp(matRandn)
    assert math.isclose(matExp[0][1], math.exp(matRandn[0][1]), rel_tol=1e-6)

    # torch.max - np.max
    maxVal = torch.max(matRandn)
    maxValExp = -1000000
    for i in range(0,2):
        for j in range(0,3):
            if matRandn[i][j] > maxValExp:
                maxValExp = matRandn[i][j]
    assert torch.eq(maxVal, maxValExp)

    # torch.sum - np.sum
    sumVal = torch.sum(matRandn)
    sumValExp = 0
    for i in range(0,2):
        for j in range(0,3):
            sumValExp = sumValExp + matRandn[i][j]
    assert sumVal == sumValExp

    # torch.matmul - np.matmul
    matA = torch.randn(1,2)
    matB = torch.randn(2,1)
    matMulRes = torch.matmul(matA, matB)
    assert torch.allclose(matMulRes[0][0], matA[0][0] * matB[0][0] + matA[0][1] * matB[1][0])

    # torch.transpose - np.transpose
    matTranspose = torch.transpose(matRandn, 0, 1)
    assert matTranspose.shape[0] == matRandn.shape[1]
    assert matTranspose.shape[1] == matRandn.shape[0]
    '''
np.random.seed
np.sqrt
np.dot
np.log
np.divide    
   '''

    # torch.manual_seed - np.random.seed
    torch.manual_seed(1)
    matSeededA = torch.randn(1,1)
    torch.manual_seed(1)
    matSeededB = torch.randn(1,1)
    assert matSeededA[0][0] == matSeededB[0][0]

    # torch.sqrt - np.sqrt
    matPos = torch.abs(matRandn)
    matSqrt = torch.sqrt(matPos)
    assert math.isclose(matSqrt[0][1], math.sqrt(matPos[0][1]), rel_tol=1e-6)

    # torch.log - np.log
    matLog = torch.log(matPos)
    assert math.isclose(matLog[0][1], math.log(matPos[0][1]), rel_tol=1e-6)

    # torch.dot - np.dot
    arrA = torch.tensor([1, 2])
    arrB = torch.tensor([4, 5])
    dotRes = torch.dot(arrA, arrB)
    assert math.isclose(dotRes, 1*4 + 2*5, rel_tol=1e-6)

    # torch.divide - np.divide
    matDivide = torch.true_divide(matRandn, 5)
    assert math.isclose(matDivide[0][1], matRandn[0][1]/5, rel_tol=1e-6)    

    
    
if __name__ == '__main__':
    test_torch_functions()
