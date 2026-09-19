#!/usr/bin/env python

import sys

import numpy as np
import torch

from aes import aes_sbox, aes_sbox_inv
from models import get_device, load_model, predict


def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]


num_classes = 256
attack_byte = 0
start_trace_to_attack = 100
number_of_traces_to_attack = 500

model_filename = "trained_model.pt"
trace_filename = "attack_traces.npz"

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_filename = sys.argv[1]
        trace_filename = sys.argv[2]

    device = get_device()
    model = load_model(model_filename, device)
    print("Input shape: " + str(model.input_shape))
    print("Device: " + str(device))

    traces = np.load(trace_filename)

    print(traces.files)

    trace_array = traces["trace_array"]
    textin_array = traces["textin_array"]
    known_keys = traces["known_keys"]

    result = predict(
        model,
        trace_array[
            start_trace_to_attack : start_trace_to_attack + number_of_traces_to_attack,
            :,
            :,
        ],
        device,
        return_logits=True,
    )
    log_probabilities = torch.log_softmax(torch.from_numpy(result), dim=1).numpy()

    log_sum_key_guess_history = np.zeros(number_of_traces_to_attack)
    log_sum_prediction = np.zeros(num_classes)

    for k in range(number_of_traces_to_attack):
        plaintext = textin_array[start_trace_to_attack + k, attack_byte]
        log_probability = log_probabilities[k]

        for label_index in range(num_classes):
            key_byte_index = aes_sbox_inv[label_index] ^ plaintext
            log_sum_prediction[key_byte_index] += log_probability[label_index]

        log_sum_key_guess_history[k] = np.argmax(log_sum_prediction)

    print("Key byte guess history:")
    print(log_sum_key_guess_history)

    print("Best key byte guess: " + str(np.argmax(log_sum_prediction)))

    print("known_keys[0]: " + str(known_keys[0]))
