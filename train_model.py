#!/usr/bin/env python

import sys

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from aes import aes_sbox
from models import cnn_best, get_device, save_model


def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]


epochs = 150
batch_size = 100
learning_rate = 0.00001
test_size = 0.2
verbose = 2
num_classes = 256
attack_byte = 0

trace_filename = "training_traces.npz"
model_filename = "trained_model.pt"

if __name__ == "__main__":
    if len(sys.argv) == 3:
        model_filename = sys.argv[1]
        trace_filename = sys.argv[2]

    traces = np.load(trace_filename)
    print(traces.files)

    trace_array = traces["trace_array"]
    textin_array = traces["textin_array"]
    known_keys = traces["known_keys"]

    # Reshape traces
    trace_array = trace_array.reshape((trace_array.shape[0], trace_array.shape[1], 1))

    number_of_traces = np.shape(trace_array)[0]
    samples_per_trace = np.shape(trace_array)[1]

    # Create model
    device = get_device()
    model = cnn_best(
        input_shape=(samples_per_trace, 1), classes=num_classes, lr=learning_rate
    ).to(device)
    print("Input shape: " + str(model.input_shape))
    print("Device: " + str(device))

    labels = np.zeros(number_of_traces)
    for x in range(number_of_traces):
        labels[x] = get_label(textin_array[x], known_keys[x], attack_byte)

    X_train, X_test, y_train, y_test = train_test_split(
        trace_array, labels, test_size=test_size
    )
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7
    )
    loss_function = nn.CrossEntropyLoss()
    model_dtype = next(model.parameters()).dtype

    for epoch in range(epochs):
        model.train()
        training_loss = 0.0
        training_correct = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=model_dtype)
            targets = targets.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            logits = model(inputs, return_logits=True)
            loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.shape[0]
            training_correct += (logits.argmax(dim=1) == targets).sum().item()

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device=device, dtype=model_dtype)
                targets = targets.to(device=device, dtype=torch.long)
                logits = model(inputs, return_logits=True)
                loss = loss_function(logits, targets)
                validation_loss += loss.item() * inputs.shape[0]
                validation_correct += (logits.argmax(dim=1) == targets).sum().item()

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {training_loss / len(train_data):.4f} - "
                f"accuracy: {training_correct / len(train_data):.4f} - "
                f"val_loss: {validation_loss / len(test_data):.4f} - "
                f"val_accuracy: {validation_correct / len(test_data):.4f}"
            )

    save_model(model, model_filename)
