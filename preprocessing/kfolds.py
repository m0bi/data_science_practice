from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

if __name__ == "__main__":
    X = pd.read_csv('kfolds.csv')
    y = X.loss
    X["kfold"] = -1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for f, (t_, v_) in enumerate(kf.split(X, y)):
        X.loc[v_, "kfold"] = f
    X.to_csv("test_5_fold.csv", index=False, sep=",")
