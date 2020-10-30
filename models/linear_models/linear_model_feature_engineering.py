import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('../../preprocessing/test_5_fold.csv')
    df = pd.get_dummies(df)
    df.to_csv('test_5_fold_OHE.csv', index=False, sep=",")
