import math
import numpy as np
import numpy.testing as npt

from dl_activate import sigmoid, tanh, softmax
from dl_data import load_csv, one_hot_encode_column, standardize_column
from dl_loss import compute_loss
from dl_model_mlp import MLPLayerConfig, MLPModel, mlp_init_weights, mlp_train

def test_MLPModel_init_1Layer_MLP():
    layer0 = MLPLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)
    npt.assert_approx_equal(len(mlp.weights),  1)
    npt.assert_approx_equal(mlp.weights[0].shape[0],  1)
    npt.assert_approx_equal(mlp.weights[0].shape[1],  3)
    npt.assert_approx_equal(len(mlp.biases),  1)
    npt.assert_approx_equal(mlp.biases[0].shape[0],  1)
    npt.assert_approx_equal(mlp.biases[0].shape[1],  1)

def test_MLPModel_init_2Layer_MLP():
    layer0 = MLPLayerConfig(5, 'tanh')
    layer1 = MLPLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    npt.assert_approx_equal(len(mlp.weights),  2)
    npt.assert_approx_equal(mlp.weights[0].shape[0],  5)
    npt.assert_approx_equal(mlp.weights[0].shape[1],  4)
    npt.assert_approx_equal(mlp.weights[1].shape[0],  3)
    npt.assert_approx_equal(mlp.weights[1].shape[1],  5 )  
    npt.assert_approx_equal(len(mlp.biases),  2)
    npt.assert_approx_equal(mlp.biases[0].shape[0],  5)
    npt.assert_approx_equal(mlp.biases[0].shape[1],  1)
    npt.assert_approx_equal(mlp.biases[1].shape[0],  3)
    npt.assert_approx_equal(mlp.biases[1].shape[1],  1)           

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
    npt.assert_approx_equal(factorHeInit,  math.sqrt(2/4))

    np.random.seed(0)    
    expectedWeightsL0 = np.random.randn(5, 4) * factorHeInit
    np.testing.assert_equal(mlp.weights[0], expectedWeightsL0)

    factorHeInit = np.sqrt(2.0/5) # Scale factor used in He initialization for layer 1
    npt.assert_approx_equal(factorHeInit,  math.sqrt(2/5))    

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
    npt.assert_approx_equal(factorHeInit,  math.sqrt(2/4))

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
    npt.assert_approx_equal(numBatches,  3)
    
    
    batchSize = 11
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)
    npt.assert_approx_equal(numBatches,  1)

def test_mlp_train_singleLayer_sigmoid_costs():
    X = np.random.randn(5, 10)
    y = np.random.randn(1, 10)

    activationFunctionID = 'sigmoid'
    layer0 = MLPLayerConfig(1, activationFunctionID)
    layers = [layer0]    
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    weightsCopy = np.copy(mlp.weights)
    biasesCopy = np.copy(mlp.biases)

    lossFunctionID = 'loss_cross_entropy'
    numEpochs = 1
    batchSize = 4
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)

    npt.assert_approx_equal(len(costs),  3)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    npt.assert_approx_equal(xBatch0.shape[1],  4)
    z0 = np.matmul(weightsCopy[0], xBatch0) + biasesCopy[0]
    a0 = sigmoid(z0)
    y_pred0 = a0
    loss0 = compute_loss(yBatch0, y_pred0, lossFunctionID)
    cost0 = np.divide(np.sum(loss0, axis = 1), loss0.shape[1])
    np.testing.assert_almost_equal(costs[0], cost0)
    np.testing.assert_equal(costs[0].shape[0], 1)


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

    weightsCopy = np.copy(mlp.weights)
    biasesCopy = np.copy(mlp.biases)
    
    lossFunctionID = 'loss_cross_entropy_softmax'
    numEpochs = 1
    batchSize = 4
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, None, batchSize, numEpochs)

    npt.assert_approx_equal(len(costs),  3)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    npt.assert_approx_equal(xBatch0.shape[1],  4)
    z0 = np.matmul(weightsCopy[0], xBatch0) + biasesCopy[0]
    a0 = tanh(z0)
    z1 = np.matmul(weightsCopy[1], a0) + biasesCopy[1]
    a1 = softmax(z1)    
    y_pred = a1
    loss0 = compute_loss(yBatch0, y_pred, lossFunctionID)
    cost0 = np.divide(np.sum(loss0, axis = 1), loss0.shape[1])
    np.testing.assert_almost_equal(costs[0], cost0)
    np.testing.assert_equal(costs[0].shape[0], 3)

def test_mlp_train():
    ''' End-to-end MLP training and evaluation test using penguin dataset found in ../data/penguins/penguins_size.csv.
    '''
    dataFilePath = '../data/penguins/penguins_size.csv'
    df = load_csv(dataFilePath)
    df = one_hot_encode_column(df, 'species')#original 'species' column is removed
    culmen_length_mean, culmean_length_stddev, df = standardize_column(df, 'culmen_length_mm')
    culmen_depth_mean, culmen_depth_stddev, df = standardize_column(df, 'culmen_depth_mm')
    flipper_length_mean, flipper_length_stddev, df = standardize_column(df, 'flipper_length_mm')
    body_mass_mean, body_mass_stddev, df = standardize_column(df, 'body_mass_g')        


    data = df.to_numpy()
    assert data.shape[0] == 344
    assert data.shape[1] == 9

    y = data[:, -3:]
    X = data[:, :6]
    

    
if __name__ == '__main__':
    dataFilePath = '../data/penguins/penguins_size.csv'
    df = load_csv(dataFilePath)
    df = one_hot_encode_column(df, 'species')
    culmen_length_mean, culmean_length_stddev, df = standardize_column(df, 'culmen_length_mm')
    culmen_depth_mean, culmen_depth_stddev, df = standardize_column(df, 'culmen_depth_mm')
    flipper_length_mean, flipper_length_stddev, df = standardize_column(df, 'flipper_length_mm')
    body_mass_mean, body_mass_stddev, df = standardize_column(df, 'body_mass_g')        


    data = df.to_numpy()
    y = data[:, -3:]    
    print(y.shape)
    print(y)
    X = data[:, :6]
    print(X)
    
