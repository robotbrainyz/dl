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
        self.weightsMomentum[layerIndex] = self.adamMomentum * self.weightsMomentum[layerIndex] + (1-self.adamMomentum) * dw
        self.biasesMomentum[layerIndex] = self.adamMomentum * self.biasesMomentum[layerIndex] + (1-self.adamMomentum) * db
        self.weightsMomentumCorrected = self.weightsMomentum[layerIndex] / (1 - self.adamMomentum**iteration)
        self.biasesMomentumCorrected = self.biasesMomentum[layerIndex] / (1 - self.adamMomentum**iteration)
                
        self.weightsScale[layerIndex] = self.adamScale * self.weightsScale[layerIndex] + (1-self.adamScale) * (dw**2)
        self.biasesScale[layerIndex] = self.adamScale * self.biasesScale[layerIndex] + (1-self.adamScale) * (db**2)
        weightsScaleCorrected = self.weightsScale[layerIndex] / (1 - self.adamScale**iteration)
        biasesScaleCorrected = self.biasesScale[layerIndex] / (1 - self.adamScale**iteration)

        epsilon = 1e-8
        weightsDelta = np.divide(self.weightsMomentumCorrected, (np.sqrt(weightsScaleCorrected) + epsilon))
        biasesDelta = np.divide(self.biasesMomentumCorrected, (np.sqrt(biasesScaleCorrected) + epsilon))
        return weightsDelta, biasesDelta

    def __init__(self, mlp, adamMomentum = 0.9, adamScale = 0.99):
        self.init(mlp, adamMomentum, adamScale)

    
        
