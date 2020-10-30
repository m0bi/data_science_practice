import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


if __name__ == "__main__":
    df = pd.read_csv('../../preprocessing/test_5_fold.csv')
    # cat1 ... cat116
    y = df.loss
    kfolds = df.kfold
    df = df.drop("kfold", axis=1)
    df = df.drop("loss", axis=1)
    df = OrdinalEncoder().fit_transform(df, y)
    df = pd.concat([pd.DataFrame(df), pd.Series(y), pd.Series(kfolds)], axis=1)
    df.to_csv('test_5_fold_FE.csv', index=False, sep=",")
