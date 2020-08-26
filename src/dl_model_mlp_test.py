import math
import numpy as np
from dl_model_mlp import MLPLayerConfig, MLPModel, mlpInitWeights

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

def test_mlpInitWeights():
    # Create a MLP with 4-5-3 nodes in the input, middle and output layer.
    # This is a 2 layer MLP (not counting the input layer).
    layer0 = MLPLayerConfig(5, 'tanh')
    layer1 = MLPLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlpInitWeights(mlp, True)

    factorHeInit = np.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert(factorHeInit == math.sqrt(2/4))

    np.random.seed(0)    
    expectedWeightsL0 = np.random.randn(5, 4) * factorHeInit
    np.testing.assert_equal(mlp.weights[0], expectedWeightsL0)

    factorHeInit = np.sqrt(2.0/5) # Scale factor used in He initialization for layer 1
    assert(factorHeInit == math.sqrt(2/5))    

    np.random.seed(0)    
    expectedWeightsL1 = np.random.randn(3, 5) * factorHeInit
    np.testing.assert_equal(mlp.weights[1], expectedWeightsL1)    

def test_mlpInitWeights1Layer():
    # Create a MLP with 4-5 nodes in the inputand output layer.
    # This is a 1 layer MLP (not counting the input layer).
    layer0 = MLPLayerConfig(5, 'softmax')
    layers = [layer0]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlpInitWeights(mlp, True)

    factorHeInit = np.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert(factorHeInit == math.sqrt(2/4))

    np.random.seed(0)    
    expectedWeightsL0 = np.random.randn(5, 4) * factorHeInit
    np.testing.assert_equal(mlp.weights[0], expectedWeightsL0)
