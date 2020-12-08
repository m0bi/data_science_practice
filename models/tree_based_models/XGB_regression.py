import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics


def run_training(X, fold):
    shift = 200
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    train = xgb.DMatrix(X_train.values, label=y_train)
    test = xgb.DMatrix(X_test.values, label=y_test)
    evallist = [(test, 'eval'), (train, 'train')]

    RANDOM_STATE = 2016
    param = {
        "objective": "reg:squarederror",
        "nthread": -1,
        "eval_metric": "mae",
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

    num_round = 2235
    bst = xgb.train(param, train, num_round, evallist,
                    early_stopping_rounds=50)

    y_pred = bst.predict(test)
    y_pred = np.exp(y_pred) - shift
    rmse = metrics.mean_absolute_error(np.exp(y_test) - shift, y_pred)
    print(f'Fold {fold}, MAE: {rmse}')


def final_run(X_train, X_test):
    shift = 200
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_train = X_train.drop("kfold", axis=1)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    train = xgb.DMatrix(X_train.values, label=y_train)
    test = xgb.DMatrix(X_test.values, label=y_test)
    evallist = [(test, 'eval'), (train, 'train')]

    RANDOM_STATE = 2016
    param = {
        "objective": "reg:squarederror",
        "nthread": -1,
        "eval_metric": "mae",
        'min_child_weight': 1,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 12,
        'subsample': 0.8,
        'alpha': 1,
        'gamma': 1,
        'silent': 1,
        'verbose_eval': True,
        'seed': RANDOM_STATE
    }

    num_round = 2235
    bst = xgb.train(param, train, num_round, evallist,
                    early_stopping_rounds=50)

    y_pred = bst.predict(test)
    y_pred = np.exp(y_pred) - shift
    rmse = metrics.mean_absolute_error(np.exp(y_test) - shift, y_pred)
    print(f'Final MAE: {rmse}')
    df = pd.DataFrame(y_pred)
    df.to_csv('./final_preds_XGB.csv', sep=",")


if __name__ == "__main__":
    #X = pd.read_csv("test_5_fold_FE.csv")
    # for i in range(5):
    #    run_training(X, i)
    X_train = pd.read_csv("test_5_fold_FE.csv")
    X_test = pd.read_csv("hold_out_FINAL.csv")
    final_run(X_train, X_test)
