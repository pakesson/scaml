#!/usr/bin/env python

import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from aes import aes_sbox
from models import cnn_best, get_device, save_model


def get_label(plaintext, key, index):
    return aes_sbox[plaintext[index] ^ key[index]]


def format_duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def print_progress(
    phase,
    batch_number,
    total_batches,
    processed_samples,
    total_loss,
    total_correct,
    start_time,
):
    update_interval = max(1, total_batches // 100)
    if batch_number != total_batches and batch_number % update_interval != 0:
        return

    elapsed = time.monotonic() - start_time
    eta = elapsed / batch_number * (total_batches - batch_number)
    end = "\n" if batch_number == total_batches else "\r"
    print(
        f"  {phase}: {batch_number}/{total_batches} batches "
        f"({batch_number / total_batches:.0%}) - "
        f"loss: {total_loss / processed_samples:.4f} - "
        f"accuracy: {total_correct / processed_samples:.4f} - "
        f"elapsed: {format_duration(elapsed)} - ETA: {format_duration(eta)}",
        end=end,
        flush=True,
    )


epochs = 150
batch_size = 100
learning_rate = 0.00001
test_size = 0.2
verbose = 2
num_classes = 256
attack_byte = 0
use_mps_bfloat16 = True

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

    number_of_traces = trace_array.shape[0]

    # Create model
    device = get_device()
    model = cnn_best(
        input_shape=trace_array.shape[1:], classes=num_classes, lr=learning_rate
    ).to(device)
    print("Input shape: " + str(model.input_shape))
    print("Device: " + str(device))
    use_amp = use_mps_bfloat16 and device.type == "mps"
    print("MPS bfloat16 AMP: " + ("enabled" if use_amp else "disabled"))

    print("Generating labels...", flush=True)
    labels = np.zeros(number_of_traces)
    for x in range(number_of_traces):
        labels[x] = get_label(textin_array[x], known_keys[x], attack_byte)

    print("Splitting training and validation data...", flush=True)
    indices = np.arange(number_of_traces)
    train_indices, test_indices = train_test_split(indices, test_size=test_size)
    dataset = TensorDataset(torch.from_numpy(trace_array), torch.from_numpy(labels))
    train_data = Subset(dataset, train_indices)
    test_data = Subset(dataset, test_indices)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    optimizer = torch.optim.RMSprop(
        model.parameters(), lr=learning_rate, alpha=0.9, eps=1e-7
    )
    loss_function = nn.CrossEntropyLoss()
    model_dtype = next(model.parameters()).dtype

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}", flush=True)
        model.train()
        training_loss = 0.0
        training_correct = 0
        training_samples = 0
        training_start = time.monotonic()
        for batch_number, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device=device, dtype=model_dtype)
            targets = targets.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
            ):
                logits = model(inputs, return_logits=True)
                loss = loss_function(logits, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.shape[0]
            training_correct += (logits.argmax(dim=1) == targets).sum().item()
            training_samples += inputs.shape[0]
            print_progress(
                "Training",
                batch_number,
                len(train_loader),
                training_samples,
                training_loss,
                training_correct,
                training_start,
            )

        model.eval()
        validation_loss = 0.0
        validation_correct = 0
        validation_samples = 0
        validation_start = time.monotonic()
        with torch.no_grad():
            for batch_number, (inputs, targets) in enumerate(test_loader, start=1):
                inputs = inputs.to(device=device, dtype=model_dtype)
                targets = targets.to(device=device, dtype=torch.long)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    logits = model(inputs, return_logits=True)
                    loss = loss_function(logits, targets)
                validation_loss += loss.item() * inputs.shape[0]
                validation_correct += (logits.argmax(dim=1) == targets).sum().item()
                validation_samples += inputs.shape[0]
                print_progress(
                    "Validation",
                    batch_number,
                    len(test_loader),
                    validation_samples,
                    validation_loss,
                    validation_correct,
                    validation_start,
                )

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"loss: {training_loss / len(train_data):.4f} - "
                f"accuracy: {training_correct / len(train_data):.4f} - "
                f"val_loss: {validation_loss / len(test_data):.4f} - "
                f"val_accuracy: {validation_correct / len(test_data):.4f}"
            )

    save_model(model, model_filename)
