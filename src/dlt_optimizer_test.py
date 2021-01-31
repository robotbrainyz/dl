import torch
from dlt_device import get_device
from dlt_model_config import NNLayerConfig
from dlt_model_mlp import MLPModel
from dlt_optimizer import AdamOptimizer

def test_adam_optimizer():
    device = get_device()
    
    layer0 = NNLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)

    momentum = 0.99
    scale = 0.9
    optimizer = AdamOptimizer(mlp, momentum, scale)
    optimizerCopy = AdamOptimizer(mlp, momentum, scale)    

    dw = torch.randn(1, 3).to(device)
    db = torch.randn(1, 1).to(device)
    iteration = 2
    layerIndex = 0
    
    weightsDelta, biasesDelta = optimizer.optimize(dw, db, iteration, layerIndex)

    expected_weights_momentum = momentum * optimizerCopy.m_weightsMomentum[layerIndex] + (1-momentum) * dw
    assert torch.allclose(optimizer.m_weightsMomentum[layerIndex], expected_weights_momentum)

    expected_biases_momentum = momentum * optimizerCopy.m_biasesMomentum[layerIndex] + (1-momentum) * db
    assert torch.allclose(optimizer.m_biasesMomentum[layerIndex], expected_biases_momentum)

    expected_weights_scale = scale * optimizerCopy.m_weightsScale[layerIndex] + (1-scale) * (dw**2)
    assert torch.allclose(optimizer.m_weightsScale[layerIndex], expected_weights_scale)

    expected_biases_scale = scale * optimizerCopy.m_biasesScale[layerIndex] + (1-scale) * (db**2)
    assert torch.allclose(optimizer.m_biasesScale[layerIndex], expected_biases_scale)

    expected_weights_momentum_corrected = expected_weights_momentum / (1 - momentum**iteration)
    weightsMomentumCorrected = optimizer.m_weightsMomentum[layerIndex] / (1 - momentum**iteration)
    assert torch.allclose(expected_weights_momentum_corrected, weightsMomentumCorrected)

    expected_biases_momentum_corrected = expected_biases_momentum / (1 - momentum**iteration)
    biasesMomentumCorrected = optimizer.m_biasesMomentum[layerIndex] / (1 - momentum**iteration)
    assert torch.allclose(expected_biases_momentum_corrected, biasesMomentumCorrected)    

    expected_weights_scale_corrected = expected_weights_scale / (1 - scale**iteration)
    weightsScaleCorrected = optimizer.m_weightsScale[layerIndex] / (1 - scale**iteration)
    assert torch.allclose(expected_weights_scale_corrected, weightsScaleCorrected)

    expected_biases_scale_corrected = expected_biases_scale / (1 - scale**iteration)
    biasesScaleCorrected = optimizer.m_biasesScale[layerIndex] / (1 - scale**iteration)
    assert torch.allclose(expected_biases_scale_corrected, biasesScaleCorrected)
    

    
