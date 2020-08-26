import logging
import numpy as np

class MLPLayerConfig:
    ''' Configuration and settings for a layer in a multi-layer perceptron model.
    '''
    def __init__(self, numNodes, activationFunctionID):
        self.numNodes = numNodes
        self.activationFunctionID = activationFunctionID # A string identifier that matches one of the functions in dl_activate.py, e.g. sigmoid.

class MLPModel:
    ''' A multi-layer perceptron model.
    '''
    
    def validateLayerConfigs(self, layerConfigs):
        ''' Validates if a valid list of layer configuration objects is provided to initialize the multi-layer perceptron model.

        Args:
            layerConfigs (list): A list of MLPLayerConfig objects.
        '''
        assert(layerConfigs is not None)
        assert(type(layerConfigs) is list)
        assert(len(layerConfigs) > 0)
        for layerConfig in layerConfigs:
            assert(type(layerConfig) is MLPLayerConfig)

    def init(self, numInputNodes, layerConfigs):
        ''' Initializes this multi-layer perceptron (MLP) model.

        Training parameters are not essential for an MLP to function and are not created on initialization.

        Args:
            numInputNodes (int): Number of input nodes in the input layer.

            layerConfigs (list): List of MLPLayerConfig objects that define each layer in this MLP.
        '''
        self.validateLayerConfigs(layerConfigs)

        self.numInputNodes = numInputNodes
        self.layerConfigs = layerConfigs # not including the input layer
        
        self.weights = []
        self.weights.append(np.zeros((layerConfigs[0].numNodes, numInputNodes)))
        if (len(layerConfigs) > 1):
            for i in range(1, len(layerConfigs)):
                self.weights.append(np.zeros((layerConfigs[i].numNodes, layerConfigs[i-1].numNodes)))

        self.biases = []
        for i in range(0, len(layerConfigs)):
            self.biases.append(np.zeros((layerConfigs[i].numNodes, 1)))
            
    def __init__(self, numInputNodes, layerConfigs):
        self.init(numInputNodes, layerConfigs)

def mlpInitWeights(mlp, useZeroSeed):
    ''' Resets the weight values in the given multi-layer perceptron using the He initialization method.

    The He initialization method is based on a paper by He et al., 2015.
    Randomly initialize weights in a layer using mean 0 and variance 1. Scale the weights by sqrt(2/(n[l-1])), where n[l-1] is the number of nodes in the previous layer.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron containing the weights to reset.

        useZeroSeed (bool): Flag to indicate if 0 is used as the seed for random number generation.
    '''
    assert(type(mlp) is MLPModel)
    assert(len(mlp.weights) > 0)

    if (useZeroSeed):
        np.random.seed(0)
        
    mlp.weights[0] = np.random.randn(mlp.weights[0].shape[0], mlp.weights[0].shape[1]) * np.sqrt(2.0 / mlp.numInputNodes)

    for i in range (1, len(mlp.weights)):
        if (useZeroSeed):
            np.random.seed(0)
        mlp.weights[i] = np.random.randn(mlp.weights[i].shape[0], mlp.weights[i].shape[1]) * np.sqrt(2.0 / mlp.weights[1].shape[1]) # Number of columns in weight matrix is the number of nodes in the previous layer.
    
def mlpSetWeights(mlp, weights):
    ''' Sets the weight values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the weights to reset.

        weights (list): A list of matrices containing the weight values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised.
    '''    
    return

def mlpSetBiases(mlp, biases):
    ''' Sets the bias values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        biases (list of matrices): A list of matrices containing the bias values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised. Each matrix in this list should only have 1 column.
    '''
    return

def mlpTrain(mlp, X, y, regularizer, batchSize=2000):
    ''' Trains the given multi-layer perceptron with the Adam optimization algorithm. Uses the given regularization parameters and batchSize for training.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

        y (matrix): A matrix where each column is a training example. The number of rows is the number of output features from the multi-layer perceptron and should match the size of the last layer in the MLP.

        regularizer (Regularizer): Object that computes the L1 or L2 regularization amount that is added to the loss during training.

        batchSize (int): If the number of columns in X is larger than batchSize, X is broken down into batches, each with batchSize number of columns for training. 
    '''
    return

def mlpPredict(mlp, X):
    ''' Performs forward propagation with input X through the given multi-layer perceptron.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

    Returns:
        matrix: Predicted output for the given input X. The number of columns is the number of examples. The number of rows is the number of output features from the MLP.
    ''' 
    return

    
    


        
