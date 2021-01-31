import math
import torch

from dlt_activate import sigmoid, tanh, softmax
from dlt_data import load_csv, one_hot_encode_column, standardize_column
from dlt_device import get_device
from dlt_loss import compute_loss, compute_cost
from dlt_model_config import NNLayerConfig
from dlt_model_mlp import MLPModel, mlp_init_weights, mlp_train, mlp_predict
from dlt_optimizer import AdamOptimizer
from dlt_regularizer import L2Regularizer

def test_MLPModel_init_1Layer_MLP():
    layer0 = NNLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)
    assert math.isclose(len(mlp.m_weights),  1, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[0].shape[0],  1, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[0].shape[1],  3, rel_tol=1e-05)
    assert math.isclose(len(mlp.m_biases),  1, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[0].shape[0],  1, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[0].shape[1],  1, rel_tol=1e-05)

def test_MLPModel_init_2Layer_MLP():
    layer0 = NNLayerConfig(5, 'tanh')
    layer1 = NNLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    assert math.isclose(len(mlp.m_weights),  2, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[0].shape[0],  5, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[0].shape[1],  4, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[1].shape[0],  3, rel_tol=1e-05)
    assert math.isclose(mlp.m_weights[1].shape[1],  5 , rel_tol=1e-05)  
    assert math.isclose(len(mlp.m_biases),  2, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[0].shape[0],  5, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[0].shape[1],  1, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[1].shape[0],  3, rel_tol=1e-05)
    assert math.isclose(mlp.m_biases[1].shape[1],  1, rel_tol=1e-05)           

def test_mlp_init_weights():
    # Create a MLP with 4-5-3 nodes in the input, middle and output layer.
    # This is a 2 layer MLP (not counting the input layer).
    device = get_device()
    
    layer0 = NNLayerConfig(5, 'tanh')
    layer1 = NNLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlp_init_weights(mlp, True)

    factorHeInit = math.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert math.isclose(factorHeInit,  math.sqrt(2/4), rel_tol=1e-05)

    torch.manual_seed(0)    
    expectedWeightsL0 = torch.randn(5, 4) * factorHeInit
    expectedWeightsL0 = expectedWeightsL0.to(device)
    assert torch.allclose(mlp.m_weights[0], expectedWeightsL0)

    factorHeInit = math.sqrt(2.0/5) # Scale factor used in He initialization for layer 1
    assert math.isclose(factorHeInit,  math.sqrt(2/5), rel_tol=1e-05)    

    torch.manual_seed(1)    
    expectedWeightsL1 = torch.randn(3, 5) * factorHeInit
    expectedWeightsL1 = expectedWeightsL1.to(device)    
    assert torch.allclose(mlp.m_weights[1], expectedWeightsL1)    

def test_mlp_init_weights1Layer():
    # Create a MLP with 4-5 nodes in the inputand output layer.
    # This is a 1 layer MLP (not counting the input layer).
    device = get_device()
    
    layer0 = NNLayerConfig(5, 'softmax')
    layers = [layer0]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    
    mlp_init_weights(mlp, True)

    factorHeInit = math.sqrt(2.0/4) # Scale factor used in He initialization for layer 0
    assert math.isclose(factorHeInit,  math.sqrt(2/4), rel_tol=1e-05)

    torch.manual_seed(0)    
    expectedWeightsL0 = torch.randn(5, 4) * factorHeInit
    expectedWeightsL0 = expectedWeightsL0.to(device)
    assert torch.allclose(mlp.m_weights[0], expectedWeightsL0)

def test_mlp_train_numBatches():
    X = torch.randn(5, 10)
    y = torch.randn(1, 10)

    activationFunctionID = 'sigmoid'
    layer0 = NNLayerConfig(1, activationFunctionID)
    layers = [layer0]    
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    lossFunctionID = 'loss_cross_entropy'
    numEpochs = 1

    batchSize = 4
    optimizer = AdamOptimizer(mlp)
    regularizationFactor = 0
    regularizer = L2Regularizer(regularizationFactor)
    
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, regularizer, optimizer, batchSize, numEpochs)
    assert math.isclose(numBatches,  3, rel_tol=1e-05)
    
    
    batchSize = 11
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, regularizer, optimizer, batchSize, numEpochs)
    assert math.isclose(numBatches,  1, rel_tol=1e-05)

def test_mlp_train_singleLayer_sigmoid_costs():
    device = get_device()    
    X = torch.randn(5, 10).to(device)
    y = torch.randn(1, 10).to(device)

    activationFunctionID = 'sigmoid'
    layer0 = NNLayerConfig(1, activationFunctionID)
    layers = [layer0]    
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    weightsCopy = []
    for weight in mlp.m_weights:
        weightsCopy.append(weight.clone().detach())
    biasesCopy = []
    for bias in mlp.m_biases:
        biasesCopy.append(bias.clone().detach())

    lossFunctionID = 'loss_cross_entropy'
    numEpochs = 1
    batchSize = 4
    optimizer = AdamOptimizer(mlp)
    regularizationFactor = 0
    regularizer = L2Regularizer(regularizationFactor)    
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, regularizer, optimizer, batchSize, numEpochs)

    assert math.isclose(len(costs),  3, rel_tol=1e-05)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    assert math.isclose(xBatch0.shape[1],  4, rel_tol=1e-05)
    z0 = torch.matmul(weightsCopy[0], xBatch0) + biasesCopy[0]
    a0 = sigmoid(z0)
    y_pred0 = a0
    loss0 = compute_loss(yBatch0, y_pred0, lossFunctionID)
    cost0 = torch.true_divide(torch.sum(loss0, dim = 1), loss0.shape[1])
    cost0 = cost0.to(device)
    assert torch.allclose(costs[0], cost0)
    assert costs[0].shape[0] == 1


def test_mlp_train_2Layer_softmax_costs():
    device = get_device()        
    X = torch.randn(5, 10).to(device)
    y = torch.randn(3, 10).to(device)

    activationFunctionID0 = 'tanh'
    activationFunctionID1 = 'softmax'    
    layer0 = NNLayerConfig(4, activationFunctionID0)
    layer1 = NNLayerConfig(3, activationFunctionID1)
    layers = [layer0, layer1]
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    weightsCopy = []
    for weight in mlp.m_weights:
        weightsCopy.append(weight.clone().detach())
    biasesCopy = []
    for bias in mlp.m_biases:
        biasesCopy.append(bias.clone().detach())
        
    lossFunctionID = 'loss_cross_entropy_softmax'
    numEpochs = 1
    batchSize = 4
    optimizer = AdamOptimizer(mlp)
    regularizationFactor = 0
    regularizer = L2Regularizer(regularizationFactor)    
    numBatches, costs = mlp_train(mlp, X, y, lossFunctionID, regularizer, optimizer, batchSize, numEpochs)

    assert math.isclose(len(costs),  3, rel_tol=1e-05)

    # Check first batch
    xBatch0 = X[:, 0:4]
    yBatch0 = y[:, 0:4]
    assert math.isclose(xBatch0.shape[1],  4, rel_tol=1e-05)
    z0 = torch.matmul(weightsCopy[0], xBatch0) + biasesCopy[0]
    a0 = tanh(z0)
    z1 = torch.matmul(weightsCopy[1], a0) + biasesCopy[1]
    a1 = softmax(z1)    
    y_pred = a1
    loss0 = compute_loss(yBatch0, y_pred, lossFunctionID)
    cost0 = torch.true_divide(torch.sum(loss0, dim = 1), loss0.shape[1])
    cost0 = cost0.to(device)
    assert torch.allclose(costs[0], cost0)
    assert costs[0].shape[0] == 3

def test_mlp_train():
    ''' End-to-end MLP training and evaluation test using penguin dataset found in ../data/penguins/penguins_size.csv.
    '''
    device = get_device()
    
    dataFilePath = '../data/penguins/penguins_size.csv'
    df = load_csv(dataFilePath)

    df = one_hot_encode_column(df, 'species')#original 'species' column is removed
    culmen_length_mean, culmean_length_stddev, df = standardize_column(df, 'culmen_length_mm')
    culmen_depth_mean, culmen_depth_stddev, df = standardize_column(df, 'culmen_depth_mm')
    flipper_length_mean, flipper_length_stddev, df = standardize_column(df, 'flipper_length_mm')
    body_mass_mean, body_mass_stddev, df = standardize_column(df, 'body_mass_g')        
    df.drop(['sex'], axis=1, inplace=True)
    df.drop(['island'], axis=1, inplace=True)
    df = df.dropna()
    data = torch.tensor(df.values)
    data.to(device)
    
    assert data.shape[0] == 342
    assert data.shape[1] == 7    

    torch.manual_seed(3)
    data=data[torch.randperm(data.size()[0])]# Shuffle to ensure distribution is random
    data.to(device)
    y = data[:, -3:] # MLP output - one-hot encoded 'species', 3 different species
    X = data[:, :4] # MLP input - culmen length, depth, flipper length, body mass
    
    X = torch.transpose(X, 0, 1)
    y = torch.transpose(y, 0, 1)
    X.to(device)
    y.to(device)
    
    assert y.shape[0] == 3
    assert X.shape[0] == 4

    trainSetSize = (data.shape[0]//5) * 4
    testSetSize = data.shape[0]-trainSetSize
    XTrain = X[:, :trainSetSize]
    yTrain = y[:, :trainSetSize]
    XTest = X[:, trainSetSize:]
    yTest = y[:, trainSetSize:]
    XTrain = XTrain.to(device)
    yTrain = yTrain.to(device)
    XTest = XTest.to(device)
    yTest = yTest.to(device)    
    assert XTrain.shape[1] == trainSetSize
    assert yTrain.shape[1] == trainSetSize
    assert XTest.shape[1] == testSetSize
    assert yTest.shape[1] == testSetSize

    activationFunctionID0 = 'tanh'
    activationFunctionID1 = 'softmax'    
    layer0 = NNLayerConfig(4, activationFunctionID0)
    layer1 = NNLayerConfig(3, activationFunctionID1)
    layers = [layer0, layer1]
    numInputNodes = X.shape[0]
    mlp = MLPModel(numInputNodes, layers)

    mlp_init_weights(mlp) # Initialize weights randomly, biases are all zeros.

    lossFunctionID = 'loss_cross_entropy_softmax'
    batchSize = 100
    numEpochs = 100
    learningRate = 0.1
    adamMomentum = 0.9
    adamScale = 0.99
    plotCosts = True
    plotTimings = True
    optimizer = AdamOptimizer(mlp, adamMomentum, adamScale)
    regularizationFactor = 0.05
    regularizer = L2Regularizer(regularizationFactor)    
    numBatches, costs = mlp_train(mlp, XTrain, yTrain, lossFunctionID, regularizer, optimizer, batchSize,numEpochs, learningRate, plotCosts, plotTimings)

    assert numBatches == 3

    trainingCostThreshold = 0.065 # Expect 93.5% training accuracy
    print(costs[-1])    
    assert torch.lt(costs[-1], trainingCostThreshold).all()

    yPred = mlp_predict(mlp, XTest)
    yPred = yPred.to(device)
    lossPred = compute_loss(yTest, yPred, lossFunctionID)
    costPred = compute_cost(lossPred) # cost is the average loss per example
    testCostThreshold = 0.1 # Expect 90% test accuracy
    print(costPred)
    assert torch.lt(costPred, testCostThreshold).all()

if __name__ == '__main__':
    test_mlp_train()
