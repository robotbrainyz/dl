import torch

from dlt_model_config import CellBasicConfig
from dlt_device import get_device
from dlt_weight_initializer import init_weights_He

class Cell:
    ''' Base class for a cell. A cell is commonly used in RNNs.
    '''

    def __init__(self):
        ''' Empty initializer for Cell base class.
        '''
        pass

    def backProp(self, params):
        ''' Base function for cell back propagation. 
        '''
        pass


class CellBasic(Cell):
    ''' Basic implementation for a cell.
    '''

    def __init__(self, cellConfig):
        self.init(cellConfig)

    def validateCellConfig(self, cellConfig):
        ''' Validates if the given cell configuration is suitable for initializing this cell.

        Args:
            cellConfig (Object): A cell configuration object.
        '''
        assert(cellConfig is not None)
        assert(type(cellConfig) is CellBasicConfig)

    def init(self, cellConfig):
        ''' Initialize basic cell.
  
        Args:
            cellConfig (CellBasicConfig): Configuration for a basic cell.
        '''        
        super().__init__()

        self.validateCellConfig(cellConfig)

        device = get_device()        

        # Configuration, weights & biases of neural networks in this cell.
        self.m_cellConfig = cellConfig

        self.m_weightsInternal = init_weights_He(cellConfig.m_layerConfigInternal.m_numNodes, cellConfig.m_numInputPrevTime + cellConfig.m_numInputPrevCellLayer)
        self.m_biasesInternal = torch.zeros((cellConfig.m_layerConfigInternal.m_numNodes, 1)).to(device)

        self.m_weightsOutput = init_weights_He(cellConfig.m_layerConfigOutput.m_numNodes, cellConfig.m_layerConfigInternal.m_numNodes)
        self.m_biasesOutput = torch.zeros((cellConfig.m_layerConfigOutput.m_numNodes, 1)).to(device)

    def backProp(self, params):
        ''' Basic cell back propagation.
        '''

        # Output backprop is only necessary if this cell produces output.
        if self.m_cellConfig.m_layerConfigOutput is not None:
            backPropOutput(params)

        # Internal backprop (through time)
        backPropInternal(params)

    def backPropOutput(self, params):
        ''' Output backpropagation for basic cell.

        Args:
            params (object): Contains derivative of output from next cell layer cell, or expected output y.
        '''
        # Output backprop:
        # Compute da. For the output part of this cell, da is also dy^, derivative of the predicted cell output y^.
        # This step is only necessary if this cell is NOT a softmax output cell, in which case we can go
        # straight to the next step to compute dz.
        
        # Output backprop:
        # Compute dz.
        # If this is a softmax ouput cell, dz = y^ -y.
        # Else dz = da * a'(z), where a' is the derivative of the output activation function for this cell.

        # Output backprop:
        # Compute dwy, dby, da(internal)
        pass

    def backPropInternal(self, params):
        # Internal backprop:
        # Compute dz(internal).
        # dz(internal) = da(internal) * a'(z), where a' is the derivative of the internal activation function for this cell.

        # Internal backprop:
        # Compute dw, db, dx.
        # The top part of dx will be the additional da(internal) for a connected cell in the previous time step.
        # The bottom part of fx will be the da(output) for a connected cell in the previous cell layer.
        pass        


class CellBasicCache:
    def __init__(self, cellConfig):
        # Cached values that facilitate back propagation.
        self.m_aCacheInternal = None
        self.m_zCacheInternal = None

        self.m_aCacheOutput = None
        self.m_zCacheOutput = None

        self.m_daCachePrevCellLayer = None
        self.m_daCachePrevTime = None        


        
        
        
        
        
        
