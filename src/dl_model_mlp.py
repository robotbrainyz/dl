import numpy as np

class MultiLayerPerceptronLayerConfig:
    ''' Configuration and settings for a layer in a multi-layer perceptron model.
    '''
    def __init__(self, numNodes, activationFunctionID):
        self.numNodes = numNodes
        self.activationFunctionID = activationFunctionID

class MultiLayerPerceptronModel:
    ''' A multi-layer perceptron model.
    '''
    
    def validateLayerConfigs(self, layerConfigs):
        ''' Validates if a valid list of layer configuration objects is provided to initialize the multi-layer perceptron model.

        Args:
            layerConfigs (list): A list of MultiLayerPerceptronLayerConfig objects.

        Returns:
            bool: True if the list of layer configuration objects is valid, False otherwise.
        ''' 
        if layerConfigs is None:
            return False
        if type(layerConfigs) is not list:
            return False
        if len(layerConfigs) == 0:
            return False
        for i in layerConfigs:
            if type(i) is not MultiLayerPerceptronLayerConfig:
                return False
        return True

    def init(self, numInputNodes, layerConfigs):
        ''' Initializes this multi-layer perceptron (MLP) model with the minimal set of parameters to function. 

        Note that training parameters are not essential for an MLP to function and are hence not created on initialization.

        Args:
            numInputNodes (int): Number of input nodes in the input layer.

            layerConfigs (list): List of MultiLayerPerceptronLayerConfig objects that define each layer in this MLP.
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

    

    


        
