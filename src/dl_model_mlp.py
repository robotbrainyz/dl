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

        Returns:
            bool: True if the list of layer configuration objects is valid, False otherwise.
        ''' 
        if layerConfigs is None:
            return False
        if type(layerConfigs) is not list:
            return False
        if len(layerConfigs) == 0:
            return False
        for layerConfig in layerConfigs:
            if type(layerConfig) is not MLPLayerConfig:
                return False
            if layerConfig.numNodes < 1:
                return False
        return True

    def init(self, numInputNodes, layerConfigs):
        ''' Initializes this multi-layer perceptron (MLP) model.

        Training parameters are not essential for an MLP to function and are not created on initialization.

        Args:
            numInputNodes (int): Number of input nodes in the input layer.

            layerConfigs (list): List of MLPLayerConfig objects that define each layer in this MLP.
        '''
        if not self.validateLayerConfigs(layerConfigs):
            return
        self.numInputNodes = numInputNodes
        self.layerConfigs = layerConfigs # not including the input layer
        
        self.weights = []
        self.weights.append(np.zeros((layerConfigs[0].numNodes, numInputNodes)))
        if (len(layerConfigs) > 1):
            for i in range(1, len(layerConfigs)):
                self.weights.append(np.zeros((layerConfigs[i].numNodes, layerConfigs[i-1].numNodes)))

        self.constants = []
        for i in range(0, len(layerConfigs)):
            self.constants.append(np.zeros((layerConfigs[i].numNodes, 1)))
            
    def __init__(self, numInputNodes, layerConfigs):
        self.init(numInputNodes, layerConfigs)

def mlpInitWeights(mlp, initWeightsMethodID):
    ''' Resets the weight values in the given multi-layer perceptron using the given weight initialization method.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron containing the weights to reset.

        initWeightsMethodID (string): 'standard', 'He', or 'Xavier'.
    '''
    return

def mlpSetWeights(mlp, weights):
    ''' Sets the weight values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the weights to reset.

        weights (list): A list of matrices containing the weight values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised.
    '''    
    return

def mlpInitConstants(mlp):
    ''' Resets the constant values in the given multi layer perceptron.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron containing the weights to reset.
    '''    
    return

def mlpSetConstants(mlp, constants):
    ''' Sets the constant values in the given multi-layer perceptron with the given list of matrices.
    
    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the constants to reset.

        constants (list of matrices): A list of matrices containing the constant values. The size of the matrices should match that in the MLP model, otherwise an exception will be raised. Each matrix in this list should only have 1 column.
    '''
    return

def mlpTrain(mlp, X, y, regularizer, batchSize=2000):
    ''' Trains the given multi-layer perceptron with the Adam optimization algorithm. Uses the given regularization parameters and batchSize for training.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the constants to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

        y (matrix): A matrix where each column is a training example. The number of rows is the number of output features from the multi-layer perceptron and should match the size of the last layer in the MLP.

        regularizer (Regularizer): Object that computes the L1 or L2 regularization amount that is added to the loss during training.

        batchSize (int): If the number of columns in X is larger than batchSize, X is broken down into batches, each with batchSize number of columns for training. 
    '''
    return

def mlpPredict(mlp, X):
    ''' Performs forward propagation with input X through the given multi-layer perceptron.

    Args:
        mlp (MLPModel): The multi-layer perceptron (MLP) containing the constants to reset.

        X (matrix): A matrix where each column is a training example. The number of rows is the number of input features to the multi-layer perceptron.

    Returns:
        matrix: Predicted output for the given input X. The number of columns is the number of examples. The number of rows is the number of output features from the MLP.
    ''' 
    return

    
    


        
