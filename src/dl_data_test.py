from dl_data import load_csv, one_hot_encode_column

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


if __name__ == "__main__":
    filePath = '../data/penguins/penguins_size.csv'
    df = load_csv(filePath)
    print(df.head())
    print(df.tail())        
    df = one_hot_encode_column(df, 'species')
    print(df.head())    
    print(df.tail())
    
