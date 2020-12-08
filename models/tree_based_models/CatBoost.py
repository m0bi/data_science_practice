from catboost import Pool, CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn import metrics


def run_training(X, fold):
    cat = []
    for i in range(115):
        cat.append(i)
        X[f'{i}'] = X[f'{i}'].astype("int")

    shift = 200

    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    train_pool = Pool(X_train, y_train, cat_features=cat)
    test_pool = Pool(X_test, y_test, cat_features=cat)
    params = {
        "loss_function": "MAE",
        "eta": 0.01,
        "depth": 12,
        "min_child_samples": 1,
        "iterations": 2000,
        "rsm": 0.5
    }
    # **PARAMS HERE
    cat = CatBoostRegressor(**params)

    cat.fit(train_pool)
    y_pred = cat.predict(test_pool)
    y_pred = np.exp(y_pred) - shift
    rmse = metrics.mean_absolute_error(np.exp(y_test) - shift, y_pred)
    print(f'Fold {fold}, MAE: {rmse}')


def final_run(X_train, X_test):
    cat = []
    for i in range(115):
        cat.append(i)
        X_train[f'{i}'] = X_train[f'{i}'].astype("int")
        X_test[f'{i}'] = X_test[f'{i}'].astype("int")

    shift = 200

    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_train = X_train.drop("kfold", axis=1)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    train_pool = Pool(X_train, y_train, cat_features=cat)
    test_pool = Pool(X_test, y_test, cat_features=cat)
    params = {
        "loss_function": "MAE",
        "eta": 0.01,
        "depth": 12,
        "min_child_samples": 1,
        "iterations": 2000,
        "rsm": 0.5
    }
    # **PARAMS HERE
    cat = CatBoostRegressor(**params)

    cat.fit(train_pool)
    y_pred = cat.predict(test_pool)
    y_pred = np.exp(y_pred) - shift
    mae = metrics.mean_absolute_error(np.exp(y_test) - shift, y_pred)
    print(f'Final MAE: {mae}')
    df = pd.DataFrame(y_pred)
    df.to_csv('./final_preds_CAT.csv')


if __name__ == "__main__":
    #X = pd.read_csv("test_5_fold_FE.csv")
    # for i in range(5):
    #    run_training(X, i)
    X_train = pd.read_csv("test_5_fold_FE.csv")
    X_test = pd.read_csv("hold_out_FINAL.csv")
    final_run(X_train, X_test)
