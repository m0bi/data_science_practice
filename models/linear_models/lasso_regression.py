import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import metrics


def run_training(X, fold):
    shift = 200
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    lasso = linear_model.Lasso()
    lasso.fit(X_train, y_train)
    preds = lasso.predict(X_test)
    preds = np.exp(preds) - shift
    score = metrics.mean_absolute_error(np.exp(y_test) - shift, preds)
    print(f'Fold {fold}, MAE {score}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_OHE.csv")
    for i in range(5):
        run_training(X, i)
