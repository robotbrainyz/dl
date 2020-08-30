import numpy as np

class L2Regularizer():
    def compute_regularization(weights, numExamples):
        ''' Returns the regularization cost and regularization terms for weight changes in each layer in the model.

        Args:
            weights (list): A List of matrices containing the weights of layers in a model.

            numExamples (int): Number of training examples.

        Returns:
            regularization_cost (float): The squared sum of weights in all layers of a model, multiplied by regularization factor, then divided by 2 x number of examples.

            regularization_weight_delta (list): A list of regularization terms for weight changes in each layer in the model. Each regularization term is added to the weight delta for a layer during back propagation.
        '''
        regularization_cost = 0
        regularization_weight_delta = []
        lambdaScaled = self.lambda/numExamples
        for layerWeights in weights:
            regularization_cost = regularization_cost + np.sum(np.square(layerWeights))
            regularization_weight_delta.append(layerWeights * lambdaScaled)
        regularization_cost = (regularization_cost * lambdaScaled) / 2
        
        return regularization_cost, regularization_weight_delta
    
    def init(lambda):
        ''' Initializes this L2 regularizer with the regularization factor lambda.

        Args:
            lambda (float): Factor to apply regularization.
        '''
        self.lambda = lambda
    
    def __init__(lambda):
        init(lambda)
