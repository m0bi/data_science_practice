import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib


def run_training(X, fold):
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = X_train.loss
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = X_test.loss
    X_test = X_test.drop("loss", axis=1)

    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    preds = regressor.predict(X_test)
    score = metrics.mean_squared_error(y_test, preds)
    print(f'Fold {fold}, RMSE {score}')


if __name__ == "__main__":
    X = pd.read_csv("test_5_fold_FE.csv")
    for i in range(5):
        run_training(X, i)
