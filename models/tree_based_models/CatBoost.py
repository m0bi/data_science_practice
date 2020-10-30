from catboost import Pool, CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn import metrics


def run_training(X, fold):
    cat = []
    for i in range(115):
        cat.append(i+1)
        X[f'{i+1}'] = X[f'{i+1}'].astype("int")

    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = X_train.loss
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = X_test.loss
    X_test = X_test.drop("loss", axis=1)

    train_pool = Pool(X_train, y_train, cat_features=cat)
    test_pool = Pool(X_test, y_test, cat_features=cat)

    cat = CatBoostRegressor(loss_function="RMSE")

    cat.fit(train_pool)
    y_pred = cat.predict(test_pool)
    rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5
    print(f'Fold {fold}, RMSE: {rmse}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_FE.csv")
    for i in range(5):
        run_training(X, i)
