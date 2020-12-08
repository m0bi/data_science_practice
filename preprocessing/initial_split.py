import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold

if __name__ == "__main__":
    X = pd.read_csv('../data/train.csv')
    y = X.loss
    X = X.drop("loss", axis=1)
    X = X.drop("id", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42)  # random state needs to stay the same to prevent leaks in our pipeline

    validation = pd.concat([X_test, y_test], axis=1).to_csv(
        "hold_out.csv", index=False, sep=",")
    folds = pd.concat([X_train, y_train], axis=1).to_csv(
        "kfolds.csv", index=False, sep=",")
