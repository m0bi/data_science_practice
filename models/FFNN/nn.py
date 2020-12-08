import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn import metrics


def build_classifier(unit1, dp1, unit2, dp2, unit3, dp3):
    nn = Sequential([
        layers.Dense(unit1, activation='relu'),
        layers.Dropout(dp1),
        layers.Dense(unit2, activation='relu'),
        layers.Dropout(dp2),
        layers.Dense(unit3, activation='relu'),
        layers.Dropout(dp3),
        layers.Dense(1)
    ])

    nn.compile(loss=tf.losses.MAE(),
               optimizer=tf.optimizers.Adam())

    return nn


def find_best_model(X):
    y = X.loss
    X = X.drop("loss", axis=1)
    X = X.drop("kfold", axis=1)
    classifier = KerasRegressor(build_fn=build_classifier)
    parameters = {
        'batch_size': [25],
        'nb_epoch': [500, 1000, 1500],
        'unit1': [128, 256],
        'dp1': [0.3],
        'unit2': [24],
        'dp2': [0.3],
        'unit3': [128, 256],
        'dp3': [0.5]
    }
    grid_search = GridSearchCV(
        classifier, param_grid=parameters, scoring="neg_root_mean_squared_error", cv=3, n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_params_)


def run_training(X, fold):
    shift = 200
    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    nn = Sequential([
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.PReLU(),
        layers.Dropout(0.6),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.PReLU(),
        layers.Dropout(0.4),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.Dense(56),
        layers.BatchNormalization(),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.Dense(1, activation=None),
    ])

    nn.compile(loss=tf.keras.losses.MeanAbsoluteError(),
               optimizer=tf.optimizers.Adam(learning_rate=0.0001), )

    # my_callbacks = [
    #    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
    # ]

    nn.fit(
        X_train,
        y_train,
        batch_size=250,
        epochs=350,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
        use_multiprocessing=True,
        workers=4,
        # callbacks=my_callbacks
    )
    # print(results.history)
    nn.save(f'./tmp/model{fold}')
    #results = nn.evaluate(X_test, y_test, batch_size=250)
    #print(results ** 0.5)

    preds = nn.predict(X_test)
    preds = np.exp(preds) - shift
    score = metrics.mean_absolute_error(np.exp(y_test) - shift, preds)
    print(f'Fold: {fold} MAE: {score}')


def final_run(X_train, X_test):
    shift = 200
    y_train = np.log(X_train.loss + shift)
    X_train = X_train.drop("loss", axis=1)
    X_train = X_train.drop("kfold", axis=1)
    y_test = np.log(X_test.loss + shift)
    X_test = X_test.drop("loss", axis=1)

    nn = Sequential([
        layers.BatchNormalization(),
        layers.Dense(1024),
        layers.PReLU(),
        layers.Dropout(0.6),
        layers.BatchNormalization(),
        layers.Dense(512),
        layers.PReLU(),
        layers.Dropout(0.4),
        layers.BatchNormalization(),
        layers.Dense(256),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(128),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(56),
        layers.PReLU(),
        layers.Dropout(0.2),
        layers.Dense(1, activation=None),
    ])

    nn.compile(loss=tf.keras.losses.MeanAbsoluteError(),
               optimizer=tf.optimizers.Adam(learning_rate=0.0001), )

    # my_callbacks = [
    #    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20),
    # ]

    nn.fit(
        X_train,
        y_train,
        batch_size=250,
        epochs=300,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
        use_multiprocessing=True,
        workers=4,
        # callbacks=my_callbacks
    )
    # print(results.history)
    nn.save(f'./tmp/final_model')

    preds = nn.predict(X_test)
    preds = np.exp(preds) - shift
    score = metrics.mean_absolute_error(np.exp(y_test) - shift, preds)
    print(f"Final MAE: {score}")
    pd_preds = pd.DataFrame(preds)
    pd_preds.to_csv('./nn_preds.csv', sep=',')


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    X = pd.read_csv("test_5_fold_SCALED.csv")
    X_test = pd.read_csv("hold_out_SCALED.csv")
    # find_best_model(X)
    # for i in range(5):
    #    run_training(X, i)
    final_run(X, X_test)
