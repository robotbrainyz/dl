
class NNLayerConfig:
    ''' Configuration and settings for a layer of nodes in a neural network.
    '''
    def init(self, numNodes, activationFunctionID):
        ''' Initializes this layer configuration.

        Args:
            numNodes (int): Number of nodes in this layer.

            activationFunctionID (string): Identifies the activation function for this layer. Needs to match one of the functions in dl_activate.py, e.g. sigmoid.
        '''
        self.m_numNodes = numNodes
        self.m_activationFunctionID = activationFunctionID
        
    def __init__(self, numNodes, activationFunctionID):
        self.init(numNodes, activationFunctionID)

class CellBasicConfig:
    ''' Configuration for a basic cell.
    '''
    def __init__(self, numInputPrevCellLayer, numInputPrevTime, layerConfigInternal, layerConfigOutput=None):
        ''' Initialize basic cell.
  
        Args:
            numInputPrevCellLayer (int): Number of inputs from a previous cell layer.

            numInputPrevTime (int): Number of inputs from a previous time step.

            layerConfigInternal (NNLayerConfig): Configuration for single layer of internal neural network nodes.

            layerConfigOutput (NNLayerConfig): Configuration for single layer of output neural network nodes.
        '''        
        self.m_numInputPrevCellLayer = numInputPrevCellLayer
        self.m_numInputPrevTime = numInputPrevTime
        self.m_layerConfigInternal = layerConfigInternal
        self.m_layerConfigOutput = layerConfigOutput

