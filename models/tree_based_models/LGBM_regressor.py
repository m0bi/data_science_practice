from sklearn import metrics
import pandas as pd
import numpy as np
import lightgbm as lgb


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

    lgb_train = lgb.Dataset(data=X_train, label=y_train,
                            categorical_feature=cat)
    lgb_valid = lgb.Dataset(
        data=X_test, label=y_test, reference=lgb_train, categorical_feature=cat)

    params = {
        'boosting_type': 'gbdt',
        'objective': "mean_absolute_error",
        'metric': "mae",
        'learning_rate': 0.01,
        'max_depth': 12,
        'min_data_in_leaf': 1,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'feature_fraction': 0.5,
        'lambda_l1': 1,
        'num_leaves': 64
    }

    gbm = lgb.train(params=params, train_set=lgb_train,
                    valid_sets=[lgb_valid], num_boost_round=3000)
    y_pred = gbm.predict(
        X_test)
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

    lgb_train = lgb.Dataset(data=X_train, label=y_train,
                            categorical_feature=cat)
    lgb_valid = lgb.Dataset(
        data=X_test, label=y_test, reference=lgb_train, categorical_feature=cat)

    params = {
        'boosting_type': 'gbdt',
        'objective': "mean_absolute_error",
        'metric': "mae",
        'learning_rate': 0.01,
        'max_depth': 12,
        'min_data_in_leaf': 1,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'feature_fraction': 0.5,
        'lambda_l1': 1,
        'num_leaves': 64
    }

    gbm = lgb.train(params=params, train_set=lgb_train,
                    valid_sets=[lgb_valid], num_boost_round=3000)
    y_pred = gbm.predict(
        X_test)
    y_pred = np.exp(y_pred) - shift
    mae = metrics.mean_absolute_error(np.exp(y_test) - shift, y_pred)
    print(f'Final MAE: {mae}')
    df = pd.DataFrame(y_pred)
    df.to_csv('./final_preds_LGBM.csv', sep=",")


if __name__ == "__main__":
    #X = pd.read_csv("test_5_fold_FE.csv")
    # for i in range(5):
    #    run_training(X, i)
    X_train = pd.read_csv("test_5_fold_FE.csv")
    X_test = pd.read_csv("hold_out_FINAL.csv")
    final_run(X_train, X_test)
