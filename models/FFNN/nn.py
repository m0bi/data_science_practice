import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd


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

    nn.compile(loss=tf.losses.MeanSquaredError(),
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

    X_train = X[X["kfold"] != fold].reset_index(drop=True)
    y_train = X_train.loss
    X_train = X_train.drop("loss", axis=1)
    X_test = X[X["kfold"] == fold].reset_index(drop=True)
    y_test = X_test.loss
    X_test = X_test.drop("loss", axis=1)

    nn = Sequential([
        layers.Dense(250, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(250, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation=None),
    ])

    nn.compile(loss=tf.losses.MeanSquaredError(),
               optimizer=tf.optimizers.Adam(learning_rate=0.0001), )

    my_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
    ]

    nn.fit(
        X_train,
        y_train,
        batch_size=250,
        epochs=500,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(X_test, y_test),
        use_multiprocessing=True,
        workers=4,
        callbacks=my_callbacks
    )
    # print(results.history)

    results = nn.evaluate(X_test, y_test, batch_size=250)
    print(results ** 0.5)

    #preds = model.predict(X_test)


if __name__ == "__main__":
    tf.keras.backend.set_floatx('float64')
    X = pd.read_csv("test_5_fold_SCALED.csv")
    # find_best_model(X)
    for i in range(5):
        run_training(X, i)
