import math
from dlt_data import load_csv, one_hot_encode_column, standardize_column

def test_load_data():
    filePath = '../data/penguins/penguins_size.csv'
    df = load_csv(filePath)
    assert df.shape[0]==344
    assert df.shape[1]==7

def test_one_hot_encode_column():
    filePath = '../data/penguins/penguins_size.csv'
    df = load_csv(filePath)
    df = one_hot_encode_column(df, 'species')

    assert df.shape[1]==9
    assert(df.loc[343]['species_Gentoo']==1)
    assert(df.loc[0]['species_Adelie']==1)

def test_standardize_column():
    filePath = '../data/penguins/penguins_size.csv'
    df = load_csv(filePath)
    columnName = 'culmen_length_mm'
    columnMean, columnStd, df = standardize_column(df, columnName)
    assert math.isclose(columnMean, 43.6665, rel_tol=1e-04)
    assert math.isclose(columnStd, 5.4595, rel_tol=1e-04)    
    assert math.isclose(df.loc[343][columnName], 1.1417, rel_tol=1e-04)
    assert math.isclose(df.loc[0][columnName], -0.83643, rel_tol=1e-04)    
