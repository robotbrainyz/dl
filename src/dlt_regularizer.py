import torch

class L2Regularizer():
    def regularize(self, weights, numExamples):
        ''' Returns the regularization cost and regularization terms for weight changes in each layer in the model.

        Args:
            weights (list): A List of matrices containing the weights of layers in a model.

            numExamples (int): Number of training examples.

        Returns:
            regularization_cost (float): The squared sum of weights in all layers of a model, multiplied by regularization factor, then divided by 2 x number of examples. This value is added to the total cost computed at each training iteration.

            regularization_weight_delta (list): A list of regularization terms for weight changes in each layer in the model. Each regularization term is added to the weight delta for a layer during back propagation.
        '''
        regularization_cost = 0
        regularization_weight_delta = []
        regularizationFactorScaled = self.regularizationFactor/numExamples
        for layerWeights in weights:
            regularization_cost = regularization_cost + torch.sum(torch.square(layerWeights))
            regularization_weight_delta.append(layerWeights * regularizationFactorScaled)
        regularization_cost = (regularization_cost * regularizationFactorScaled) / 2
        
        return regularization_cost, regularization_weight_delta
    
    def init(self, regularizationFactor):
        ''' Initializes this L2 regularizer with the regularization factor.

        Args:
            regularizationFactor (float): Factor to apply regularization.
        '''
        self.regularizationFactor = regularizationFactor
    
    def __init__(self, regularizationFactor):
        self.init(regularizationFactor)
