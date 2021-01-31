import math
import torch
from dlt_weight_initializer import init_weights_He

def test_init_weights_He():
    torch.manual_seed(0)
    numInputs = 5
    numNodes = 7
    weights = init_weights_He(numInputs, numNodes)
    assert weights.shape[0] == 5
    assert weights.shape[1] == 7
    assert math.isclose(weights[1][3], 0.18707, rel_tol=1e-04)    
    assert math.isclose(weights[4][6], -0.6415, rel_tol=1e-04)
