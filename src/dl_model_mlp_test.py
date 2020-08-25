from dl_model_mlp import MLPLayerConfig, MLPModel


def test_MLPModel_invalidConfig():
    # Test that a 'None' or null MLP layer configs is not accepted.        
    layers = None
    numInputNodes = 1
    mlp = MLPModel(numInputNodes, layers)

    mlpWeightsExist = True
    mlpConstantsExist = True
    
    try:
        len(mlp.weights)
    except:
        mlpWeightsExist = False
    assert not mlpWeightsExist

    try:
        len(mlp.constants)
    except:
        mlpConstantsExist = False
    assert not mlpConstantsExist

    # Test that an empty list of MLP layer configs is not accepted.    
    layers = []
    mlp = MLPModel(numInputNodes, layers)

    mlpWeightsExist = True
    mlpConstantsExist = True
    
    try:
        len(mlp.weights)
    except:
        mlpWeightsExist = False
    assert not mlpWeightsExist

    try:
        len(mlp.constants)
    except:
        mlpConstantsExist = False
    assert not mlpConstantsExist

    # Test that non-list type for MLP layer configs is not accepted.
    layers = 1
    mlp = MLPModel(numInputNodes, layers)

    mlpWeightsExist = True
    mlpConstantsExist = True
    
    try:
        len(mlp.weights)
    except:
        mlpWeightsExist = False
    assert not mlpWeightsExist

    try:
        len(mlp.constants)
    except:
        mlpConstantsExist = False
    assert not mlpConstantsExist

    # Test that a non MLPLayerConfig object is not accepted.
    layers = [MLPLayerConfig(2, 'sigmoid'), 'non MLPLayerConfig string']
    mlp = MLPModel(numInputNodes, layers)

    mlpWeightsExist = True
    mlpConstantsExist = True
    
    try:
        len(mlp.weights)
    except:
        mlpWeightsExist = False
    assert not mlpWeightsExist

    try:
        len(mlp.constants)
    except:
        mlpConstantsExist = False
    assert not mlpConstantsExist        

    
def test_MLPModel_init_1Layer_MLP():
    layer0 = MLPLayerConfig(1, 'sigmoid')
    layers = [layer0]
    numInputNodes = 3
    mlp = MLPModel(numInputNodes, layers)
    assert len(mlp.weights) == 1
    assert mlp.weights[0].shape[0] == 1
    assert mlp.weights[0].shape[1] == 3
    assert len(mlp.constants) == 1
    assert mlp.constants[0].shape[0] == 1        
    assert mlp.constants[0].shape[1] == 1

def test_MLPModel_init_2Layer_MLP():
    layer0 = MLPLayerConfig(5, 'tanh')
    layer1 = MLPLayerConfig(3, 'softmax')
    layers = [layer0, layer1]
    numInputNodes = 4
    mlp = MLPModel(numInputNodes, layers)
    assert len(mlp.weights) == 2
    assert mlp.weights[0].shape[0] == 5
    assert mlp.weights[0].shape[1] == 4
    assert mlp.weights[1].shape[0] == 3
    assert mlp.weights[1].shape[1] == 5   
    assert len(mlp.constants) == 2
    assert mlp.constants[0].shape[0] == 5
    assert mlp.constants[0].shape[1] == 1
    assert mlp.constants[1].shape[0] == 3
    assert mlp.constants[1].shape[1] == 1            
