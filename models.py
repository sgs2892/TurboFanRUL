"""
models.py
Holds all the different models to be used for testing

authors: Kunal Nayyar & Srujan Shetty
"""

from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional


def modelBi(sequence_length, nb_features, nb_out, r2_keras):
    # The first layer is an Dense layer with 128 units followed by a Bi-directional LSTM layer with 128 units.
    # Dropout is also applied after LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    model = Sequential()
    model.add(Dense(
        input_shape=(sequence_length, nb_features),
        units=128))
    model.add(Bidirectional(LSTM(
        units=128,
        return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae', r2_keras])
    return model


def modelLstm128(sequence_length, nb_features, nb_out, r2_keras):
    # The first layer is an LSTM layer with 128 units followed by another LSTM layer with 128 units.
    # Dropout is also applied after each LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    model = Sequential()
    model.add(LSTM(
        input_shape=(sequence_length, nb_features),
        units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=128,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae', r2_keras])
    return model


def modelSingleLstm256(sequence_length, nb_features, nb_out, r2_keras):
    # The first layer is an Dense layer with 128 units followed by another LSTM layer with 128 units.
    # Dropout is also applied after LSTM layer to control overfitting.
    # Final layer is a Dense output layer with single unit and linear activation since this is a regression problem.
    model = Sequential()
    model.add(Dense(
        input_shape=(sequence_length, nb_features),
        units=128))
    model.add(Dropout(0.2))
    model.add(LSTM(
        units=256,
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=nb_out))
    model.add(Activation("linear"))
    model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae', r2_keras])
    return model
