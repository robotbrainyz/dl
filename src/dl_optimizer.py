import numpy as np

class DummyOptimizer:
    def optimize(self, dw, db, iteration, layerIndex):
        pass
    
    def __init__(self, mlp):
        pass


class AdamOptimizer:
    ''' Adam optimizer
    '''
    def init(self, mlp, adamMomentum, adamScale):
        ''' Initializes this AdamOptimizer with an MLP model.

        Assume that the MLP model is already initialized.

        Args:
            mlp (MLPModel): Instance of initialized MLPModel.

            adamMomentum (float): Proportion of previous derivatives to retain vs the latest derivatives computed. A value between 0 to 1.

            adamScale (float): Proportion to previous derivative used to scale the latest derivatives computed. A value between 0 to 1.
        '''
        self.adamMomentum = adamMomentum
        self.adamScale = adamScale
        self.weightsMomentum = []
        self.weightsScale = []
        self.biasesMomentum = []
        self.biasesScale = []
        for layerWeights in mlp.weights:
            self.weightsMomentum.append(np.zeros(layerWeights.shape))
            self.weightsScale.append(np.zeros(layerWeights.shape))
        for layerBiases in mlp.biases:
            self.biasesMomentum.append(np.zeros(layerBiases.shape))
            self.biasesScale.append(np.zeros(layerBiases.shape))

    def optimize(self, dw, db, iteration, layerIndex):
        ''' Computes the change in weights and biases given the derivative of the weights and biases with respect to the loss, and the layer index in the multi-layer perceptron.

        Args:
            dw (matrix): A (n[l] * n[l-1]) matrix that contains the derivatives of the weights in the given layer. n[l] is the number of nodes in the given layer, and n[l-1] is the number of nodes in the previous layer.

            db (matrix): A (n[l] * 1) matrix that contains the derivatives of the biases in the given layer. n[l] is the number of nodes in the given layer.

            iteration (int): Training iteration that increases as training progresses. This value is used to reduce the effect of this Adam optimizer as training progresses further.

            layerIndex (int): Index of the layer to identify the weights and biases momentum and scales in this optimizer.

        Returns:
            weightsDelta (matrix): A (n[l] * n[l-1]) matrix that contains the recommended change in the weights for the given layer in the multi-layer perceptron.

            biasesDelta (matrix): A (n[l] * 1) matrix that contains the recommended change in the biases for the given layer in the multi-layer perceptron.
        '''
        momentumInverse = 1 - self.adamMomentum
        momentumInverseIter = (1 - self.adamMomentum**iteration)
        self.weightsMomentum[layerIndex] = self.adamMomentum * self.weightsMomentum[layerIndex] + momentumInverse * dw
        self.biasesMomentum[layerIndex] = self.adamMomentum * self.biasesMomentum[layerIndex] + momentumInverse * db
        weightsMomentumCorrected = self.weightsMomentum[layerIndex] / momentumInverseIter
        biasesMomentumCorrected = self.biasesMomentum[layerIndex] / momentumInverseIter

        scaleInverse = 1 - self.adamScale
        scaleInverseIter = 1 - self.adamScale**iteration
        self.weightsScale[layerIndex] = self.adamScale * self.weightsScale[layerIndex] + scaleInverse * (dw**2)
        self.biasesScale[layerIndex] = self.adamScale * self.biasesScale[layerIndex] + scaleInverse * (db**2)
        weightsScaleCorrected = self.weightsScale[layerIndex] / scaleInverseIter
        biasesScaleCorrected = self.biasesScale[layerIndex] / scaleInverseIter

        epsilon = 1e-8
        weightsDelta = np.divide(weightsMomentumCorrected, (np.sqrt(weightsScaleCorrected) + epsilon))
        biasesDelta = np.divide(biasesMomentumCorrected, (np.sqrt(biasesScaleCorrected) + epsilon))
        return weightsDelta, biasesDelta

    def __init__(self, mlp, adamMomentum = 0.9, adamScale = 0.99):
        self.init(mlp, adamMomentum, adamScale)

    
        
