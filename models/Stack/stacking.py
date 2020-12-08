import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb


def combine_csv(*argv):
    dfs = [pd.read_csv(arg) for arg in argv]
    df_final = pd.DataFrame()
    for i in range(len(dfs)):
        df_final = pd.concat([df_final, dfs[i]['0']], axis=1)
    return df_final


def training(X_train, X_test, y_train, y_test):
    lgb_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_valid = lgb.Dataset(
        data=X_test, label=y_test, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': "mean_absolute_error",
        'metric': "mae",
        'learning_rate': 0.01,
        'max_depth': 6,
        'min_data_in_leaf': 1,
        'lambda_l1': 1,
        'num_leaves': 18
    }

    gbm = lgb.train(params=params, train_set=lgb_train,
                    valid_sets=[lgb_valid], num_boost_round=1000)
    y_pred = gbm.predict(X_test)
    y_pred = np.exp(y_pred)
    mae = metrics.mean_absolute_error(np.exp(y_test), y_pred)
    print(f'Final MAE: {mae}')


if __name__ == "__main__":
    df = combine_csv('./final_preds_XGB.csv', './final_preds_LGBM.csv',
                     './final_preds_CAT.csv', './nn_preds.csv')
    cols = pd.Series(["XGB", "LGBM", "CAT", "NN"])
    df.columns = cols
    df["Distance"] = df.max(axis=1) - df.min(axis=1)
    kmeans = KMeans(n_clusters=5)
    df["kmeans"] = kmeans.fit_predict(df)
    df[["XGB"]] = np.log(df[["XGB"]])
    df[["LGBM"]] = np.log(df[["LGBM"]])
    df[["CAT"]] = np.log(df[["CAT"]])
    df[["NN"]] = np.log(df[["NN"]])
    df[["Distance"]] = np.log(df[["Distance"]])
    y = pd.read_csv('../../preprocessing/hold_out.csv')
    y = y.loss
    y = np.log(y)
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.1, shuffle=True, random_state=42)
    training(X_train, X_test, y_train, y_test)
