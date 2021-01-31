import math
import time
import torch

from dlt_back import back_softmax, back, back_linear
from dlt_device import get_device
from dlt_forward import forward
from dlt_loss import compute_loss, compute_cost, loss_cross_entropy_back
from dlt_model_config import NNLayerConfig
from dlt_plot import plot_costs, plot_time
from dlt_weight_initializer import init_weights_He

class MLPModel:
    ''' A multi-layer perceptron model.
    '''
    
    def validateLayerConfigs(self, layerConfigs):
        ''' Validates if a valid list of layer configuration objects is provided to initialize the multi-layer perceptron model.

        Args:
            layerConfigs (list): A list of NNLayerConfig objects.
        '''
        assert(layerConfigs is not None)
        assert(type(layerConfigs) is list)
        assert(len(layerConfigs) > 0)
        for layerConfig in layerConfigs:
            assert(type(layerConfig) is NNLayerConfig)

    def init(self, numInputNodes, layerConfigs):
        ''' Initializes this multi-layer perceptron (MLP) model.

        Training parameters are not essential for an MLP to function and are not created on initialization.

        Args:
            numInputNodes (int): Number of input nodes in the input layer.

            layerConfigs (list): List of NNLayerConfig objects that define each layer in this MLP.
        '''
        device = get_device()
    
        self.validateLayerConfigs(layerConfigs)

        self.m_numInputNodes = numInputNodes
        self.m_layerConfigs = layerConfigs # not including the input layer
        
        self.m_weights = []
        self.m_weights.append(torch.zeros((layerConfigs[0].m_numNodes, self.m_numInputNodes)).to(device))
        if (len(layerConfigs) > 1):
            for i in range(1, len(layerConfigs)):
                self.m_weights.append(torch.zeros((layerConfigs[i].m_numNodes, layerConfigs[i-1].m_numNodes)).to(device))

        self.m_biases = []
        for i in range(0, len(layerConfigs)):
            self.m_biases.append(torch.zeros((layerConfigs[i].m_numNodes, 1)).to(device))
            
    def __init__(self, numInputNodes, layerConfigs):
        self.init(numInputNodes, layerConfigs)

def mlp_init_weights(mlp, useSeeds=False):
    ''' Resets the weight values in the given multi-layer perceptron using the He initialization method.

    The He initialization method is based on a paper by He et al., 2015.
    Randomly initialize weights in a layer using mean 0 and variance 1. Scale the weights by sqrt(2/(n[l-1])), where n[l-1] is the number of nodes in the previous layer.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron containing the weights to reset.

        useSeeds (bool): Flag to indicate if seeds are used for random number generation. This is useful in testing where the same set of random numbers can be generated again to validate against the weight values.
    '''
    assert(type(mlp) is MLPModel)
    assert(len(mlp.m_weights) > 0)

    device = get_device()

    if (useSeeds):
        torch.manual_seed(0)
        
    mlp.m_weights[0] = init_weights_He(mlp.m_weights[0].shape[0], mlp.m_weights[0].shape[1])

    for i in range (1, len(mlp.m_weights)):
        if (useSeeds):
            torch.manual_seed(i)
        mlp.m_weights[i] = init_weights_He(mlp.m_weights[i].shape[0], mlp.m_weights[i].shape[1])
    
def mlp_set_weights(mlp, weights):
    ''' Sets the weight values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the weights to reset.

        weights (list): A list of matrices containing the weight values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised.
    '''    
    return

def mlp_set_biases(mlp, biases):
    ''' Sets the bias values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        biases (list of matrices): A list of matrices containing the bias values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised. Each matrix in this list should only have 1 column.
    '''
    return

def mlp_train(mlp, X, y, lossFunctionID, regularizer, optimizer, batchSize=2000, numEpochs=1, learningRate = 0.1, plotCosts = False, plotTimings = False):
    ''' Trains the given multi-layer perceptron for 1 epoch with the Adam optimization algorithm. 1 epoch propagates all training examples through the multi-layer perceptron exactly once. Uses the given regularization parameters and batchSize for training.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron. This function assumes that X is already pre-processed for training, i.e. any data set shuffling, etc. is already done.

        y (matrix): A matrix where each column is a training example. The number of rows is the number of output features from the multi-layer perceptron and should match the size of the last layer in the MLP.

        lossFunctionID (string): Function name that matches a loss function in dl_loss.py.

        regularizer (Regularizer): Object that computes the L1 or L2 regularization amount that is added to the loss during training.

        batchSize (int): If the number of columns in X is larger than batchSize, X is broken down into batches, each with batchSize number of columns for training. 

        numEpochs (int): Number of epochs to train the given MLP. 1 epoch propagates all training examples through the multi-layer perceptron exactly once.

        learningRate (float): Scalar multiplied against weight derivatives before subtracting derivatives from weights.

        plotCosts (bool): Flag to specify if the cost values per iteration are plotted as a visual graph at the end of the training.

    Returns:
        numBatches (int): Number of batches given the batchSize and number of training examples.
    '''
    # Number of input nodes should be the same as the number of rows in the input training data matrix
    assert(mlp.m_weights[0].shape[1] == X.shape[0])

    # Number of output nodes should be the same as the number of rows in the training data expected output.
    assert(mlp.m_weights[len(mlp.m_weights)-1].shape[0] == y.shape[0])
    
    assert(batchSize > 0)
    assert(X.shape[1] == y.shape[1])

    device = get_device()
    
    numBatches = X.shape[1]//batchSize + 1
    costs = [] # List of computed costs (average loss) per batch

    totalTime = 0
    totalTimings = []
    batchTimings = []
    
    iteration = 1
    for epochIndex in range(0, numEpochs):
        for batchIndex in range(0, numBatches):
            startTime = time.time()
            
            # Select the columns for this batch from input X, and output y
            startColumn = batchIndex * batchSize
            endColumn = min((batchIndex + 1) * batchSize, X.shape[1])
            XBatch = X[:, startColumn:endColumn]
            yBatch = y[:, startColumn:endColumn]
            XBatch = XBatch.to(device)
            yBatch = yBatch.to(device)
        
            # Forward propagate
            aCache = [] # Cache to contain activation output of all layers
            zCache = [] # Cache to contain input to activation for all layers
            for layerIndex in range(0, len(mlp.m_layerConfigs)):
                if layerIndex > 0:
                    layerInput = aCache[layerIndex-1]
                else:
                    layerInput = XBatch
                z, a = forward(layerInput,
                              mlp.m_weights[layerIndex],
                              mlp.m_biases[layerIndex],
                              mlp.m_layerConfigs[layerIndex].m_activationFunctionID)
                aCache.append(a.to(device))
                zCache.append(z.to(device))
    
            # Compute Loss and cost
            yBatch_pred = aCache[len(aCache)-1] # Predicted output is activation output of last layer
            loss = compute_loss(yBatch, yBatch_pred, lossFunctionID)
            regCost, regWeightsDelta = regularizer.regularize(mlp.m_weights, loss.shape[1])
            costs.append(compute_cost(loss+regCost)) # cost is the average loss per example

            # Back propagate
            if (mlp.m_layerConfigs[len(mlp.m_layerConfigs)-1].m_activationFunctionID != 'softmax'):
                da = loss_cross_entropy_back(yBatch, yBatch_pred)
            else:
                da = None
                
            for layerIndex in range(len(mlp.m_layerConfigs)-1, -1, -1):
                layerActivationFunctionID = mlp.m_layerConfigs[layerIndex].m_activationFunctionID
                if (layerActivationFunctionID != 'softmax'):
                    dz = back(da, zCache[layerIndex], layerActivationFunctionID)
                else:
                    dz = back_softmax(yBatch, yBatch_pred)
                if layerIndex == 0:
                    aPrev = XBatch
                else:
                    aPrev = aCache[layerIndex-1]
                dw, db, da = back_linear(dz, aPrev, mlp.m_weights[layerIndex]) # Replace da to continue back prop in previous layer.

                weightsDelta, biasesDelta = optimizer.optimize(dw, db, iteration, layerIndex)
                mlp.m_weights[layerIndex] = mlp.m_weights[layerIndex] - learningRate * (weightsDelta + regWeightsDelta[layerIndex])
                mlp.m_biases[layerIndex] = mlp.m_biases[layerIndex] - learningRate * biasesDelta
                iteration = iteration + 1
            endTime = time.time()
            elapsed = endTime-startTime
            totalTime = totalTime + elapsed
            batchTimings.append(elapsed)        
            totalTimings.append(totalTime)            
    if plotCosts:
        plot_costs(costs)
    if plotTimings:
        plot_time(totalTime, totalTimings, batchTimings)
    return numBatches, costs

def mlp_predict(mlp, X):
    ''' Performs forward propagation with input X through the given multi-layer perceptron.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

    Returns:
        matrix: Predicted output for the given input X. The number of columns is the number of examples. The number of rows is the number of output features from the MLP.
    '''
    aCache = [] # Cache to contain activation output of all layers
    for layerIndex in range(0, len(mlp.m_layerConfigs)):
        if layerIndex > 0:
            layerInput = aCache[layerIndex-1]
        else:
            layerInput = X
        z, a = forward(layerInput,
                       mlp.m_weights[layerIndex],
                       mlp.m_biases[layerIndex],
                       mlp.m_layerConfigs[layerIndex].m_activationFunctionID)
        aCache.append(a)
    
    # Predicted output is activation output of last layer
    return aCache[len(aCache)-1] 
