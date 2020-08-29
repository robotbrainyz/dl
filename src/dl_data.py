import os
import pandas as pd

def load_csv(filePath):
    ''' Loads a csv file.

    Args:
        filePath (string): Path to csv file.

    Returns:
        data (DataFrame): Loaded data.
    '''
    assert os.path.exists(filePath)
    return pd.read_csv(filePath)

def one_hot_encode_column(df, columnName):
    ''' One-hot encodes a column in the dataFrame.
  
    Removes original column with categorical values.

    Args:
        df (DataFrame): Data containing a matrix of values.

        columnName (string): Name of column to perform one-hot encoding.        

    Returns:
        data (DataFrame): Data frame with One-hot encoded column. Returned data has the original column with categorical values removed.
    '''
    assert (columnName in df.columns)
    df = pd.concat([df,pd.get_dummies(df[columnName], prefix=columnName)],axis=1)
    return df.drop([columnName],axis=1, inplace=False)

def standardize_column(df, columnName):
    ''' Standardize values in the given dataFrame for the given column.

    This function is not optimal for large data sets. For large data sets, the column mean and standard deviation should be computed separately, and then used to standardize column values in divided batches of the data.

    Args:
        df (DataFrame): Data containing a matrix of values.

        columnName (string): Name of column to perform one-hot encoding.            

    Returns:
        columnMean (float): Mean value of column values.

        columnStd (float): Standard deviation of column values.

        data (DataFrame): Data frame with standardized values in the specified column.
    '''
    assert (columnName in df.columns)
    columnSum = df[columnName].sum()
    rowCount = df.shape[0]
    columnMean = columnSum/rowCount
    columnStdDev = df[columnName].std()
    df[columnName] = (df[columnName]-columnMean)/columnStdDev
    return columnMean, columnStdDev, df
