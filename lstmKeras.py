"""
lstmKeras.py
The driver program that calls the models.py & dataprep.py files

authors: Kunal Nayyar & Srujan Shetty
"""


import dataprep
import models
import tensorflow as tf
import keras.backend as K
from keras.models import  load_model
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

# define path to save model
model_path = './regression_model.h5'


# Data Ingestion

train_df = dataprep.get_train_data()
test_df = dataprep.get_test_data()


# Data Preprocessing

# pick a large window size of 50 cycles
sequence_length = 50


# function to reshape features into (samples, time steps, features)
def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    """
    :param id_df: The trajectories pertaining to the engine id
    :param seq_length: The length of the sequence
    :param seq_cols: The generated RUL of each trajectory
    
    :yields: a sequence of len seq_length.
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]

    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


# pick the feature columns
sensor_cols = ['s' + str(i) for i in range(1, 22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(train_df[train_df['id'] == id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

# generate sequences and convert to numpy array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
print(seq_array.shape)


# function to generate labels
def gen_labels(id_df, seq_length, label):
    """
    :param id_df: The trajectories pertaining to the engine id
    :param seq_length: The length of the sequence
    :param label: The generated RUL of each trajectory
    :return: Matrix of RULs (has shape SequenceLength * 1)
    """

    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length:num_elements, :]


# generate labels
label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
             for id in train_df['id'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)
print("label array")
print(label_array)

# Modeling


def r2_keras(y_true, y_pred):
    """
    Coefficient of Determination
    """
    res = K.sum(K.square(y_true - y_pred))
    tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - res / (tot + K.epsilon())


nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

modelList = [models.modelBi(sequence_length, nb_features, nb_out, r2_keras), models.modelLstm128(sequence_length, nb_features, nb_out, r2_keras),
             models.modelSingleLstm256(sequence_length, nb_features, nb_out, r2_keras)]

for model in modelList:
    print(model.summary())

    # fit the network
    history = model.fit(seq_array, label_array, epochs=100, batch_size=200, validation_split=0.05, verbose=2,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0,
                                                                 mode='min'),
                                   keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True,
                                                                   mode='min', verbose=0)])

    # list all data in history
    print(history.history.keys())

    # summarize history for MAE
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model MAE')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    # summarize history for Loss
    fig_acc = plt.figure(figsize=(10, 10))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()

    # training metrics
    scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)
    print('\nMAE: {}'.format(scores[1]))
    print('\nR^2: {}'.format(scores[2]))

    y_pred = model.predict(seq_array, verbose=1, batch_size=200)
    y_true = label_array

    test_set = pd.DataFrame(y_pred)

    # EVALUATE ON TEST DATA

    # We pick the last sequence for each id in the test data
    seqArray = [test_df[test_df['id'] == id][sequence_cols].values[-sequence_length:]
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]

    seqArray = np.asarray(seqArray).astype(np.float32)

    # Picking the labels

    y_vals = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    labelArray = test_df.groupby('id')['RUL'].nth(-1)[y_vals].values
    labelArray = labelArray.reshape(labelArray.shape[0], 1).astype(np.float32)

    # if best iteration's model was saved then load and use it
    if os.path.isfile(model_path):
        estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

        # test metrics
        scores_test = estimator.evaluate(seqArray, labelArray, verbose=2)
        print('\nMAE: {}'.format(scores_test[1]))
        print('\nR^2: {}'.format(scores_test[2]))

        y_pred_test = estimator.predict(seqArray)
        y_true_test = labelArray

        # Plot in blue color the predicted data and in green color the
        # actual data to verify visually the accuracy of the model.
        fig_verify = plt.figure(figsize=(100, 50))
        plt.plot(y_pred_test, color="blue")
        plt.plot(y_true_test, color="green")
        plt.title('prediction')
        plt.ylabel('value')
        plt.xlabel('row')
        plt.legend(['predicted', 'actual data'])
        plt.show()

        # keras.backend.clear_session()

