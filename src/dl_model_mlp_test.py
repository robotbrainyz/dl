import math
import numpy as np

from dl_activate import sigmoid, tanh, softmax
from dl_loss import compute_loss
from dl_model_mlp import MLPLayerConfig, MLPModel, mlp_init_weights, mlp_train

def test_MLPModel_init_1Layer_MLP():
    layer0 = MLPLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)
    assert len(mlp.weights) == 1
    assert mlp.weights[0].shape[0] == 1
    assert mlp.weights[0].shape[1] == 3
    assert len(mlp.biases) == 1
    assert mlp.biases[0].shape[0] == 1        
    assert mlp.biases[0].shape[1] == 1

def test_MLPModel_init_2Layer_MLP():
    layer0 = MLPLayerConfig(5, 'tanh')
    layer1 = MLPLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    assert len(mlp.weights) == 2
    assert mlp.weights[0].shape[0] == 5
    assert mlp.weights[0].shape[1] == 4
    assert mlp.weights[1].shape[0] == 3
    assert mlp.weights[1].shape[1] == 5   
    assert len(mlp.biases) == 2
    assert mlp.biases[0].shape[0] == 5
    assert mlp.biases[0].shape[1] == 1
    assert mlp.biases[1].shape[0] == 3
    assert mlp.biases[1].shape[1] == 1            

def test_mlp_init_weights():
    # Create a MLP with 4-5-3 nodes in the input, middle and output layer.
    # This is a 2 layer MLP (not counting the input layer).
    layer0 = MLPLayerConfig(5, 'tanh')
    layer1 = MLPLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlp_init_weights(mlp, True)

    factorHeInit = np.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert(factorHeInit == math.sqrt(2/4))

    np.random.seed(0)    
    expectedWeightsL0 = np.random.randn(5, 4) * factorHeInit
    np.testing.assert_equal(mlp.weights[0], expectedWeightsL0)

    factorHeInit = np.sqrt(2.0/5) # Scale factor used in He initialization for layer 1
    assert(factorHeInit == math.sqrt(2/5))    

    np.random.seed(1)    
    expectedWeightsL1 = np.random.randn(3, 5) * factorHeInit
    np.testing.assert_equal(mlp.weights[1], expectedWeightsL1)    

def test_mlp_init_weights1Layer():
    # Create a MLP with 4-5 nodes in the inputand output layer.
    # This is a 1 layer MLP (not counting the input layer).
    layer0 = MLPLayerConfig(5, 'softmax')
    layers = [layer0]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlp_init_weights(mlp, True)

    factorHeInit = np.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert(factorHeInit == math.sqrt(2/4))

    np.random.seed(0)    
    expectedWeightsL0 = np.random.randn(5, 4) * factorHeInit
    np.testing.assert_equal(mlp.weights[0], expectedWeightsL0)

def test_mlp_train_numBatches():
    X = np.random.randn(5, 10)
    y = np.random.randn(1, 10)

    activationFunctionID = 'sigmoid'
    layer0 = MLPLayerConfig(1, activationFunctionID)
    layers = [layer0]    
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    lossFunctionID = 'loss_cross_entropy'
    numEpochs = 1

    batchSize = 4
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)
    assert(numBatches == 3)
    
    
    batchSize = 11
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)
    assert(numBatches == 1)

def test_mlp_train_singleLayer_sigmoid_costs():
    X = np.random.randn(5, 10)
    y = np.random.randn(1, 10)

    activationFunctionID = 'sigmoid'
    layer0 = MLPLayerConfig(1, activationFunctionID)
    layers = [layer0]    
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    lossFunctionID = 'loss_cross_entropy'
    numEpochs = 1
    batchSize = 4
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)

    assert(len(costs) == 3)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    assert(xBatch0.shape[1] == 4)
    z0 = np.matmul(mlp.weights[0], xBatch0) + mlp.biases[0]
    a0 = sigmoid(z0)
    y_pred0 = a0
    loss0 = compute_loss(yBatch0, y_pred0, lossFunctionID)
    cost0 = np.divide(np.sum(loss0, axis = 1), loss0.shape[1])
    np.testing.assert_almost_equal(costs[0], cost0)
    np.testing.assert_equal(costs[0].shape[0], 1)


    # Check last batch
    xBatch2 = X[:, 8:10]
    yBatch2 = y[:, 8:10]
    assert(xBatch2.shape[1] == 2)
    z2 = np.matmul(mlp.weights[0], xBatch2) + mlp.biases[0]
    a2 = sigmoid(z2)
    y_pred2 = a2
    loss2 = compute_loss(yBatch2, y_pred2, lossFunctionID)
    cost2 = np.divide(np.sum(loss2, axis = 1), loss2.shape[1])
    np.testing.assert_almost_equal(costs[2], cost2)
    np.testing.assert_equal(costs[2].shape[0], 1)    

def test_mlp_train_2Layer_softmax_costs():
    X = np.random.randn(5, 10)
    y = np.random.randn(3, 10)

    activationFunctionID0 = 'tanh'
    activationFunctionID1 = 'softmax'    
    layer0 = MLPLayerConfig(4, activationFunctionID0)
    layer1 = MLPLayerConfig(3, activationFunctionID1)
    layers = [layer0, layer1]
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    lossFunctionID = 'loss_cross_entropy_softmax'
    numEpochs = 1
    batchSize = 4
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)

    assert(len(costs) == 3)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    assert(xBatch0.shape[1] == 4)
    z0 = np.matmul(mlp.weights[0], xBatch0) + mlp.biases[0]
    a0 = tanh(z0)
    z1 = np.matmul(mlp.weights[1], a0) + mlp.biases[1]
    a1 = softmax(z1)    
    y_pred = a1
    loss0 = compute_loss(yBatch0, y_pred, lossFunctionID)
    cost0 = np.divide(np.sum(loss0, axis = 1), loss0.shape[1])
    np.testing.assert_almost_equal(costs[0], cost0)
    np.testing.assert_equal(costs[0].shape[0], 3)

    # Check last batch
    xBatch2 = X[:, 8:10]
    yBatch2 = y[:, 8:10]
    assert(xBatch2.shape[1] == 2)
    z0 = np.matmul(mlp.weights[0], xBatch2) + mlp.biases[0]
    a0 = tanh(z0)
    z1 = np.matmul(mlp.weights[1], a0) + mlp.biases[1]
    a1 = softmax(z1)    
    y_pred = a1
    loss2 = compute_loss(yBatch2, y_pred, lossFunctionID)
    cost2 = np.divide(np.sum(loss2, axis = 1), loss2.shape[1])
    np.testing.assert_almost_equal(costs[2], cost2)
    np.testing.assert_equal(costs[2].shape[0], 3)    
    
