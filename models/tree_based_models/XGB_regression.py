import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics


def run_training(X, fold):

    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = X_train.loss
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = X_test.loss
    X_test = X_test.drop("loss", axis=1)

    train = xgb.DMatrix(X_train.values, label=y_train.values)
    test = xgb.DMatrix(X_test.values, label=y_test.values)
    evallist = [(test, 'eval'), (train, 'train')]
    param = {
        "objective": "reg:squarederror",
        "nthread": -1,
        "eval_metric": "rmse"
    }
    num_round = 100
    bst = xgb.train(param, train, num_round, evallist,
                    early_stopping_rounds=50)

    y_pred = bst.predict(test)
    rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5
    print(f'Fold {fold}, RMSE: {rmse}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_FE.csv")
    for i in range(5):
        run_training(X, i)
