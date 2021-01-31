import math
import torch

from dlt_device import get_device

def init_weights_He(numNodes, numInputs):
    ''' Creates a weight matrix for a neural network layer using the number of nodes in the layer and the number of inputs to the layer.
    '''
    device = get_device()    
    return torch.randn(numNodes, numInputs).to(device) * math.sqrt(2.0 / numInputs)
