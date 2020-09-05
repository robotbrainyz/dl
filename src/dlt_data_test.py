import math
from dl_data import load_csv, one_hot_encode_column, standardize_column

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
    math.isclose(columnMean, 43.6665)
    math.isclose(columnStd, 5.4595)    
    math.isclose(df.loc[343][columnName], 1.1417)
    math.isclose(df.loc[0][columnName], -0.83643)    
