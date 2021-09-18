#!/usr/bin/env python

import sys
import math
import numpy as np

from tensorflow.keras.models import load_model

from aes import aes_sbox, aes_sbox_inv

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]

num_classes = 256
attack_byte = 0
start_trace_to_attack = 100
number_of_traces_to_attack = 500

model_filename = "trained_model.h5"
trace_filename = "attack_traces.npz"
sensitivity_map_filename = "sensitivity_map.npz"

def apply_occlusion(sample, x, occlusion_size=1, occlusion_value=0):
    occluded_sample = np.array(sample, copy=True)
    occluded_sample[x:x+occlusion_size, :] = occlusion_value
    return occluded_sample

def get_occlusion_sensitivity(sample, model, class_index, occlusion_size=1):
    print("Generating occlusion sensitivity maps...")

    sensitivity_map = np.zeros(math.ceil(sample.shape[0] / occlusion_size))

    occlusions = [
        apply_occlusion(sample, x, occlusion_size, 0.5)
        for x in range(0, sample.shape[0], occlusion_size)
    ]

    predictions = model.predict(np.array(occlusions)) # Set batch_size?
    target_class_predictions = [
        prediction[class_index] for prediction in predictions
    ]

    for x, confidence in zip(range(sensitivity_map.shape[0]), target_class_predictions):
        sensitivity_map[x] = 1.0-confidence

    # Scale back up
    result = np.zeros(sample.shape[0])
    for x in range(result.shape[0]):
        result[x] = sensitivity_map[x // occlusion_size]

    return result

def explain(data, model, class_index, occlusion_size=10):
    sensitivity_maps = []

    # Make sure the data shape is (num_traces, num_points_per_trace, x)
    if len(data.shape) == 2:
        data = data.reshape((1, data.shape[0], data.shape[1]))
    elif len(data.shape) != 3:
        raise ValueError("unsupported data shape")

    sensitivity_maps = [
        get_occlusion_sensitivity(data[x], model, class_index, occlusion_size)
        for x in range(data.shape[0])
    ]

    return sensitivity_maps

if __name__ == '__main__':
    if len(sys.argv) == 4:
        model_filename = sys.argv[1]
        trace_filename = sys.argv[2]
        sensitivity_map_filename = sys.argv[3]

    model = load_model(model_filename)
    print("Input shape: " + str(model.input_shape))

    traces = np.load(trace_filename)

    print(traces.files)

    trace_array = traces['trace_array']
    textin_array = traces['textin_array']
    known_keys = traces['known_keys']

    trace_array = trace_array.reshape((trace_array.shape[0], trace_array.shape[1], 1))

    # Run an initial prediction before we try to explain anything
    result = model.predict(trace_array[start_trace_to_attack:start_trace_to_attack+number_of_traces_to_attack, :, :])

    log10_sum_prediction = np.zeros(num_classes)
    for k in range(number_of_traces_to_attack):
        plaintext = textin_array[start_trace_to_attack+k, attack_byte]
        prediction = result[k]
        for l in range(num_classes):
            key_byte_index = (aes_sbox_inv[l] ^ plaintext)
            log10_sum_prediction[key_byte_index] += np.log10(prediction[l] + 1e-22)

    print("Best key byte guess: " + str(np.argmax(log10_sum_prediction)))
    print("known_keys[0]: " + str(known_keys[0]))

    # Run explainer
    data = trace_array[start_trace_to_attack, :, :]
    key_index = np.argmax(log10_sum_prediction)
    class_index = aes_sbox[textin_array[start_trace_to_attack, attack_byte] ^ key_index]
    occlusion_size = 1
    sensitivity_map = explain(data, model, class_index, occlusion_size)
    sensitivity_map = sensitivity_map[0]

    # Save results
    np.savez_compressed(sensitivity_map_filename, sensitivity_map=sensitivity_map)

    # Visualize the results
    fig = plt.figure()
    plt.title(f"Occlusion sensitivity for key byte {attack_byte} in trace {start_trace_to_attack}")
    ax = fig.gca()
    x = np.linspace(0, sensitivity_map.shape[0]-1, sensitivity_map.shape[0])
    for i in range(0, sensitivity_map.shape[0]-1, occlusion_size):
        color = sensitivity_map[i]/max(sensitivity_map)
        ax.plot(x[i:i+occlusion_size+1], data[i:i+occlusion_size+1, 0], color=plt.cm.plasma(color))
    plt.show()
