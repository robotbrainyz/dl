import torch

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
        if torch.cuda.is_available():  
            dev = "cuda:0"
        else:  
            dev = "cpu"
        device = torch.device(dev)
    
        self.m_adamMomentum = adamMomentum
        self.m_adamScale = adamScale
        self.m_weightsMomentum = []
        self.m_weightsScale = []
        self.m_biasesMomentum = []
        self.m_biasesScale = []
        for layerWeights in mlp.m_weights:
            self.m_weightsMomentum.append(torch.zeros(layerWeights.shape).to(device))
            self.m_weightsScale.append(torch.zeros(layerWeights.shape).to(device))
        for layerBiases in mlp.m_biases:
            self.m_biasesMomentum.append(torch.zeros(layerBiases.shape).to(device))
            self.m_biasesScale.append(torch.zeros(layerBiases.shape).to(device))

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
        momentumInverse = 1 - self.m_adamMomentum
        momentumInverseIter = (1 - self.m_adamMomentum**iteration)
        self.m_weightsMomentum[layerIndex] = self.m_adamMomentum * self.m_weightsMomentum[layerIndex] + momentumInverse * dw
        self.m_biasesMomentum[layerIndex] = self.m_adamMomentum * self.m_biasesMomentum[layerIndex] + momentumInverse * db
        weightsMomentumCorrected = self.m_weightsMomentum[layerIndex] / momentumInverseIter
        biasesMomentumCorrected = self.m_biasesMomentum[layerIndex] / momentumInverseIter

        scaleInverse = 1 - self.m_adamScale
        scaleInverseIter = 1 - self.m_adamScale**iteration
        self.m_weightsScale[layerIndex] = self.m_adamScale * self.m_weightsScale[layerIndex] + scaleInverse * (dw**2)
        self.m_biasesScale[layerIndex] = self.m_adamScale * self.m_biasesScale[layerIndex] + scaleInverse * (db**2)
        weightsScaleCorrected = self.m_weightsScale[layerIndex] / scaleInverseIter
        biasesScaleCorrected = self.m_biasesScale[layerIndex] / scaleInverseIter

        epsilon = 1e-8
        weightsDelta = torch.true_divide(weightsMomentumCorrected, (torch.sqrt(weightsScaleCorrected) + epsilon))
        biasesDelta = torch.true_divide(biasesMomentumCorrected, (torch.sqrt(biasesScaleCorrected) + epsilon))
        return weightsDelta, biasesDelta

    def __init__(self, mlp, adamMomentum = 0.9, adamScale = 0.99):
        self.init(mlp, adamMomentum, adamScale)

    
        
