import numpy as np
import numpy.testing as npt
from dl_model_mlp import MLPModel, MLPLayerConfig
from dl_optimizer import AdamOptimizer

def test_adam_optimizer():
    layer0 = MLPLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)

    momentum = 0.99
    scale = 0.9
    optimizer = AdamOptimizer(mlp, momentum, scale)
    optimizerCopy = AdamOptimizer(mlp, momentum, scale)    

    dw = np.random.randn(1, 3)
    db = np.random.randn(1, 1)
    iteration = 2
    layerIndex = 0
    
    weightsDelta, biasesDelta = optimizer.optimize(dw, db, iteration, layerIndex)

    expected_weights_momentum = momentum * optimizerCopy.weightsMomentum[layerIndex] + (1-momentum) * dw
    npt.assert_array_almost_equal_nulp(optimizer.weightsMomentum[layerIndex], expected_weights_momentum)

    expected_biases_momentum = momentum * optimizerCopy.biasesMomentum[layerIndex] + (1-momentum) * db
    npt.assert_array_almost_equal_nulp(optimizer.biasesMomentum[layerIndex], expected_biases_momentum)

    expected_weights_scale = scale * optimizerCopy.weightsScale[layerIndex] + (1-scale) * (dw**2)
    npt.assert_array_almost_equal_nulp(optimizer.weightsScale[layerIndex], expected_weights_scale)

    expected_biases_scale = scale * optimizerCopy.biasesScale[layerIndex] + (1-scale) * (db**2)
    npt.assert_array_almost_equal_nulp(optimizer.biasesScale[layerIndex], expected_biases_scale)

    expected_weights_momentum_corrected = expected_weights_momentum / (1 - momentum**iteration)
    weightsMomentumCorrected = optimizer.weightsMomentum[layerIndex] / (1 - momentum**iteration)
    npt.assert_array_almost_equal_nulp(expected_weights_momentum_corrected, weightsMomentumCorrected)

    expected_biases_momentum_corrected = expected_biases_momentum / (1 - momentum**iteration)
    biasesMomentumCorrected = optimizer.biasesMomentum[layerIndex] / (1 - momentum**iteration)
    npt.assert_array_almost_equal_nulp(expected_biases_momentum_corrected, biasesMomentumCorrected)    

    expected_weights_scale_corrected = expected_weights_scale / (1 - scale**iteration)
    weightsScaleCorrected = optimizer.weightsScale[layerIndex] / (1 - scale**iteration)
    npt.assert_array_almost_equal_nulp(expected_weights_scale_corrected, weightsScaleCorrected)

    expected_biases_scale_corrected = expected_biases_scale / (1 - scale**iteration)
    biasesScaleCorrected = optimizer.biasesScale[layerIndex] / (1 - scale**iteration)
    npt.assert_array_almost_equal_nulp(expected_biases_scale_corrected, biasesScaleCorrected)
    

    
