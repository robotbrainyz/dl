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

    def __init__(self):
        ''' Initialize basic cell.
        '''        
        super().__init__()

        weightsInternal = None
        biasesInternal = None
        activationFunctionIDInternal = None

        aCacheInternal = None
        zCacheInternal = None
        
        weightsOutput = None
        biasesOutput = None
        activationFunctionIDOutput = None        

        aCacheOutput = None
        zCacheOutput = None

        daPrevLayer = None
        daPrevTime = None

    def backProp(self, params):
        ''' Basic cell back propagation.
        '''

        # Output backprop is only necessary if this cell produces output.
        if weightsOutput is not None:
            backPropOutput(params)

        # Internal backprop (through time)
        backPropInternal(params)

    def backPropOutput(self, params):
        ''' Output backpropagation for basic cell.

        Args:
            params (object): Contains derivative of output from next layer cell, or expected output y.
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
        # The bottom part of fx will be the da(output) for a connected cell in the previous layer.
        pass        


        
        
        
        
        
        
        
