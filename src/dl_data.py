import os
import pandas as pd

def load_csv(filePath):
    ''' Loads a csv file.

    Args:
        filePath (string): Path to csv file.

    Returns:
        Data (DataFrame): Loaded data.
    '''
    assert os.path.exists(filePath)
    return pd.read_csv(filePath)

def one_hot_encode_column(dataFrame, columnName):
    ''' One-hot encodes a column in the dataFrame.

    Args:
        dataFrame (DataFrame): Data containing a matrix of values.

        columnName (string): Name of column to perform one-hot encoding.        

    '''
    dataFrame = pd.concat([dataFrame,pd.get_dummies(dataFrame[columnName], prefix=columnName)],axis=1)
    return dataFrame.drop([columnName],axis=1, inplace=False)
