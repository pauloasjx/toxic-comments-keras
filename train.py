from utils import ToxicComments

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def train(model, X_train, y_train, batch_size, epochs):

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, train_size=0.90
    )

    filepath = "best_model.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )

    early = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    callbacks_list = [checkpoint, early]

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_validation, y_validation),
        callbacks=callbacks_list,
        verbose=1,
    )

    return model


def test(model, X_test):
    y_pred = model.predict(X_test, batch_size=1024, verbose=1)

    return y_pred
