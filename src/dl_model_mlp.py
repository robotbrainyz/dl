import logging
import numpy as np

from dl_back import back_softmax, back, back_linear
from dl_forward import forward
from dl_loss import compute_loss, compute_cost, loss_cross_entropy_back
from dl_plot import plot_costs

class MLPLayerConfig:
    ''' Configuration and settings for a layer in a multi-layer perceptron model.
    '''
    def init(self, numNodes, activationFunctionID):
        ''' Initializes this layer configuration.

        Args:
            numNodes (int): Number of nodes in this layer.

            activationFunctionID (string): Identifies the activation function for this layer. Needs to match one of the functions in dl_activate.py, e.g. sigmoid.
        '''
        self.numNodes = numNodes
        self.activationFunctionID = activationFunctionID
        
    def __init__(self, numNodes, activationFunctionID):
        self.init(numNodes, activationFunctionID)

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

def mlp_init_weights(mlp, useSeeds=False):
    ''' Resets the weight values in the given multi-layer perceptron using the He initialization method.

    The He initialization method is based on a paper by He et al., 2015.
    Randomly initialize weights in a layer using mean 0 and variance 1. Scale the weights by sqrt(2/(n[l-1])), where n[l-1] is the number of nodes in the previous layer.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron containing the weights to reset.

        useSeeds (bool): Flag to indicate if seeds are used for random number generation. This is useful in testing where the same set of random numbers can be generated again to validate against the weight values.
    '''
    assert(type(mlp) is MLPModel)
    assert(len(mlp.weights) > 0)

    if (useSeeds):
        np.random.seed(0)
        
    mlp.weights[0] = np.random.randn(mlp.weights[0].shape[0], mlp.weights[0].shape[1]) * np.sqrt(2.0 / mlp.numInputNodes)

    for i in range (1, len(mlp.weights)):
        if (useSeeds):
            np.random.seed(i)
        mlp.weights[i] = np.random.randn(mlp.weights[i].shape[0], mlp.weights[i].shape[1]) * np.sqrt(2.0 / mlp.weights[1].shape[1]) # Number of columns in weight matrix is the number of nodes in the previous layer.
    
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

def mlp_train(mlp, X, y, lossFunctionID, regularizer, batchSize=2000, numEpochs=1, learningRate = 0.1, adamMomentum = 0.9, adamScale = 0.99, plotCosts = False):
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

        adamMomentum (float): Proportion of previous derivatives to retain vs the latest derivatives computed. A value between 0 to 1.

        adamScale (float): Proportion to previous derivative used to scale the latest derivatives computed. A value between 0 to 1.

    Returns:
        numBatches (int): Number of batches given the batchSize and number of training examples.
    '''
    # Number of input nodes should be the same as the number of rows in the input training data matrix
    assert(mlp.weights[0].shape[1] == X.shape[0])

    # Number of output nodes should be the same as the number of rows in the training data expected output.
    assert(mlp.weights[len(mlp.weights)-1].shape[0] == y.shape[0])
    
    assert(batchSize > 0)
    assert(X.shape[1] == y.shape[1])
    
    numBatches = X.shape[1]//batchSize + 1
    costs = [] # List of computed costs (average loss) per batch

    # Adam optimizer - Momentum and scale for weights and biases, iteration count
    iteration = 1
    weightsMomentum = []
    weightsScale = []
    biasesMomentum = []
    biasesScale = []
    for layerWeights in mlp.weights:
        weightsMomentum.append(np.zeros(layerWeights.shape))
        weightsScale.append(np.zeros(layerWeights.shape))
    for layerBiases in mlp.biases:
        biasesMomentum.append(np.zeros(layerBiases.shape))
        biasesScale.append(np.zeros(layerBiases.shape))                

    for epochIndex in range(0, numEpochs):
        for batchIndex in range(0, numBatches):
            # Select the columns for this batch from input X, and output y
            startColumn = batchIndex * batchSize
            endColumn = min((batchIndex + 1) * batchSize, X.shape[1])
            XBatch = X[:, startColumn:endColumn]
            yBatch = y[:, startColumn:endColumn]
        
            # Forward propagate
            aCache = [] # Cache to contain activation output of all layers
            zCache = [] # Cache to contain input to activation for all layers
            for layerIndex in range(0, len(mlp.layerConfigs)):
                if layerIndex > 0:
                    layerInput = aCache[layerIndex-1]
                else:
                    layerInput = XBatch
                z, a = forward(layerInput,
                              mlp.weights[layerIndex],
                              mlp.biases[layerIndex],
                              mlp.layerConfigs[layerIndex].activationFunctionID)
                aCache.append(a)
                zCache.append(z)
    
            # Compute Loss and cost
            yBatch_pred = aCache[len(aCache)-1] # Predicted output is activation output of last layer
            loss = compute_loss(yBatch, yBatch_pred, lossFunctionID)
            costs.append(compute_cost(loss)) # cost is the average loss per example
            
            # Back propagate
            if (mlp.layerConfigs[len(mlp.layerConfigs)-1].activationFunctionID != 'softmax'):
                da = loss_cross_entropy_back(yBatch, yBatch_pred)
            else:
                da = None
                
            for layerIndex in range(len(mlp.layerConfigs)-1, -1, -1):
                layerActivationFunctionID = mlp.layerConfigs[layerIndex].activationFunctionID
                if (layerActivationFunctionID != 'softmax'):
                    dz = back(da, zCache[layerIndex], layerActivationFunctionID)
                else:
                    dz = back_softmax(yBatch, yBatch_pred)
                if layerIndex == 0:
                    aPrev = XBatch
                else:
                    aPrev = aCache[layerIndex-1]
                dw, db, da = back_linear(dz, aPrev, mlp.weights[layerIndex]) # Replace da to continue back prop in previous layer.

                weightsMomentum[layerIndex] = adamMomentum * weightsMomentum[layerIndex] + (1-adamMomentum) * dw
                biasesMomentum[layerIndex] = adamMomentum * biasesMomentum[layerIndex] + (1-adamMomentum) * db
                weightsMomentumCorrected = weightsMomentum[layerIndex] / (1 - adamMomentum**iteration)
                biasesMomentumCorrected = biasesMomentum[layerIndex] / (1 - adamMomentum**iteration)
                
                weightsScale[layerIndex] = adamScale * weightsScale[layerIndex] + (1-adamScale) * (dw**2)
                biasesScale[layerIndex] = adamScale * biasesScale[layerIndex] + (1-adamScale) * (db**2)
                weightsScaleCorrected = weightsScale[layerIndex] / (1 - adamScale**iteration)
                biasesScaleCorrected = biasesScale[layerIndex] / (1 - adamScale**iteration)

                epsilon = 1e-8
                weightsDelta = np.divide(weightsMomentumCorrected, (np.sqrt(weightsScaleCorrected) + epsilon))
                biasesDelta = np.divide(biasesMomentumCorrected, (np.sqrt(biasesScaleCorrected) + epsilon))
                mlp.weights[layerIndex] = mlp.weights[layerIndex] - learningRate * weightsDelta
                mlp.biases[layerIndex] = mlp.biases[layerIndex] - learningRate * biasesDelta
                iteration = iteration + 1

    if plotCosts:
        plot_costs(costs)
    return numBatches, costs

def mlp_predict(mlp, X):
    ''' Performs forward propagation with input X through the given multi-layer perceptron.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the biases to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

    Returns:
        matrix: Predicted output for the given input X. The number of columns is the number of examples. The number of rows is the number of output features from the MLP.
    '''

    '''
    aPrev = None
    for layerIndex in range(0, len(mlp.layerConfigs)):
        if layerIndex > 0:
            layerInput = aPrev
        else:
            layerInput = X
        aPrev = forward(layerInput,
                        mlp.weights[layerIndex],
                        mlp.biases[layerIndex],
                        mlp.layerConfigs[layerIndex].activationFunctionID)
    return aPrev

    '''
    return
    


        
