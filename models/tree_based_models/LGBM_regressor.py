from sklearn import metrics
import pandas as pd
import numpy as np
import lightgbm as lgb


def run_training(X, fold):
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = X_train.loss
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = X_test.loss
    X_test = X_test.drop("loss", axis=1)

    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_valid = lgb.Dataset(
        data=X_test, label=y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': "regression"
    }

    gbm = lgb.train(params=params, train_set=lgb_train,
                    valid_sets=[lgb_valid], num_boost_round=1000)
    y_pred = gbm.predict(
        X_test)
    rmse = metrics.mean_squared_error(y_test, y_pred) ** 0.5
    print(f'Fold {fold}, RMSE: {rmse}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_FE.csv")
    for i in range(5):
        run_training(X, i)
