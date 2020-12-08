import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib
import numpy as np


def run_training(X, fold):
    shift = 200
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    preds = regressor.predict(X_test)
    preds = np.exp(preds) - shift
    score = metrics.mean_absolute_error(np.exp(y_test) - shift, preds)
    print(f'Fold {fold}, MAE {score}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_FE.csv")
    for i in range(5):
        run_training(X, i)
