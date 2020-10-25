#!/usr/bin/env python

import sys
import numpy as np

from keras.models import load_model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from models import cnn_best
from aes import aes_sbox, aes_sbox_inv

def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]

num_classes = 256
attack_byte = 0
start_trace_to_attack = 100
number_of_traces_to_attack = 500

model_filename = "trained_model.h5"
trace_filename = "attack_traces.npz"

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model_filename = sys.argv[1]
        trace_filename = sys.argv[2]

    model = load_model(model_filename)
    print("Input shape: " + str(model.input_shape))

    traces = np.load(trace_filename)

    print(traces.files)

    trace_array = traces['trace_array']
    textin_array = traces['textin_array']
    known_keys = traces['known_keys']

    trace_array = trace_array.reshape((trace_array.shape[0], trace_array.shape[1], 1))

    result = model.predict(trace_array[start_trace_to_attack:start_trace_to_attack+number_of_traces_to_attack, :, :])

    log10_sum_key_guess_history = np.zeros(number_of_traces_to_attack)
    log10_sum_prediction = np.zeros(num_classes)

    for k in range(number_of_traces_to_attack):
        plaintext = textin_array[start_trace_to_attack+k, attack_byte]
        prediction = result[k]

        for l in range(num_classes):
            key_byte_index = (aes_sbox_inv[l] ^ plaintext)
            log10_sum_prediction[key_byte_index] += np.log10(prediction[l] + 1e-22)

        log10_sum_key_guess_history[k] = np.argmax(log10_sum_prediction)

    #print("Key byte guess history:")
    #print(log10_sum_key_guess_history)

    print("Best key byte guess: " + str(np.argmax(log10_sum_prediction)))

    print("known_keys[0]: " + str(known_keys[0]))