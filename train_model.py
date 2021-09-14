#!/usr/bin/env python

import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models import cnn_best
from aes import aes_sbox

def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]

epochs = 150
batch_size = 100
learning_rate = 0.00001
test_size = 0.2
verbose = True
num_classes = 256
attack_byte = 0

trace_filename = "training_traces.npz"
model_filename = "trained_model.h5"

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model_filename = sys.argv[1]
        trace_filename = sys.argv[2]

    traces = np.load(trace_filename)
    print(traces.files)

    trace_array = traces['trace_array']
    textin_array = traces['textin_array']
    known_keys = traces['known_keys']

    # Reshape traces
    trace_array = trace_array.reshape((trace_array.shape[0], trace_array.shape[1], 1))

    number_of_traces = np.shape(trace_array)[0]
    samples_per_trace = np.shape(trace_array)[1]

    # Create model
    model = cnn_best(input_shape=(samples_per_trace,1), classes=num_classes, lr=learning_rate)
    print("Input shape: " + str(model.input_shape))

    labels = np.zeros(number_of_traces)
    for x in range(number_of_traces):
        labels[x] = get_label(textin_array[x], known_keys[x], attack_byte)

    labels = to_categorical(labels, num_classes=num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        trace_array, labels, test_size=test_size)

    history = model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_test, y_test),
        )

    model.save(model_filename)
