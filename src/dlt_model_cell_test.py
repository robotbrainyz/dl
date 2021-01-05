import pytest

from dlt_model_cell import Cell, CellBasic

class CellForTest:
    ''' Test class for a cell in an RNN.
    '''

    def __init__(self, value):
        ''' Initialize test class definition of RNN cell.
        '''
        self.value = value

    def backProp(self, params):
        ''' Back propagation function for tests.
        '''
        self.value = self.value + params.valueA

class CellChildForTest(CellForTest):
    ''' Test class that inherits TestCell.
    '''

    def __init__(self, value, valueB):
        ''' Initialize test class definition of RNN cell.
        '''
        super().__init__(value)
        self.valueB = valueB

    def backProp(self, params):
        ''' Back propagation function for tests.
        '''
        self.value = self.value + params.valueA
        self.valueB = self.valueB + params.valueB

class ParamsForTest:
    ''' Test class holding back-prop parameters for Cell.
    '''

    def __init__(self, valueA):
        self.valueA = valueA

class ParamsBForTest:
    ''' Test class holding back-prop parameters for another type of Cell.
    '''

    def __init__(self, valueA, valueB):
        self.valueA = valueA
        self.valueB = valueB
        
    
def test_RNNCell_references():
    listCells = []
    cellA = CellForTest(1)
    cellB = CellForTest(2)
    listCells.append(cellA)
    listCells.append(cellB)
    cellA.value = 3
    assert listCells[0].value == 3 # Check if value of cell in list is modified when cell is modified directly.
    listCells[1].value = 4
    assert cellB.value == 4 # Check if cell value is modified when cell in list is modified.

def test_RNNCell_functionInheritance():
    cellA = CellForTest(1)
    cellB = CellChildForTest(2, 3)
    listCells = [cellA, cellB]
    paramsA = ParamsForTest(4)
    paramsB = ParamsBForTest(5, 6)
    listCells[0].backProp(paramsA)

    # Check if single value parameter is used by the invokation of backProp of
    # CellForTest to increment its variable value.    
    assert listCells[0].value == 5 

    listCells[1].backProp(paramsB)

    # Check if double value parameter is used by the invokation of overriden backProp of
    # CellChildForTest to increment both of its variable values.    
    assert listCells[1].value == 7 
    assert listCells[1].valueB == 9

    # Check that an error is raised if a parameter object with the wrong parameter constituents is used for backprop,
    with pytest.raises(Exception):
        listCells[1].backProp(paramsA)

       
