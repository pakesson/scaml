import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CNNBest(nn.Module):
    def __init__(self, input_shape=(1, 5000), classes=256):
        super().__init__()
        self.input_shape = tuple(input_shape)
        self.classes = classes

        self.block1_conv1 = nn.Conv1d(self.input_shape[0], 64, 11, padding="same")
        self.block1_pool = nn.AvgPool1d(2, stride=2)
        self.block2_conv1 = nn.Conv1d(64, 128, 11, padding="same")
        self.block2_pool = nn.AvgPool1d(2, stride=2)
        self.block3_conv1 = nn.Conv1d(128, 256, 11, padding="same")
        self.block3_pool = nn.AvgPool1d(2, stride=2)
        self.block4_conv1 = nn.Conv1d(256, 512, 11, padding="same")
        self.block4_pool = nn.AvgPool1d(2, stride=2)
        self.block5_conv1 = nn.Conv1d(512, 512, 11, padding="same")
        self.block5_pool = nn.AvgPool1d(2, stride=2)

        pooled_length = self.input_shape[1]
        for _ in range(5):
            pooled_length //= 2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * pooled_length, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.predictions = nn.Linear(4096, classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.apply(self._initialize_layer)

    @staticmethod
    def _initialize_layer(layer):
        if isinstance(layer, (nn.Conv1d, nn.Linear)):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs, return_logits=False):
        x = self.block1_pool(self.relu(self.block1_conv1(inputs)))
        x = self.block2_pool(self.relu(self.block2_conv1(x)))
        x = self.block3_pool(self.relu(self.block3_conv1(x)))
        x = self.block4_pool(self.relu(self.block4_conv1(x)))
        x = self.block5_pool(self.relu(self.block5_conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.predictions(x)
        if return_logits:
            return x
        return self.softmax(x)


# From ASCAD
# https://github.com/ANSSI-FR/ASCAD/blob/master/ASCAD_train_models.py#L38
# License: "The databases, the Deep Learning models and the companion python
# scripts of this repository are placed into the public domain."
def cnn_best(input_shape=(1, 5000), classes=256, lr=0.00001):
    # The learning rate is accepted here to preserve the existing model factory API.
    del lr
    return CNNBest(input_shape=input_shape, classes=classes)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_model(model, filename, epoch=None, training_config=None):
    checkpoint = {
        "format_version": 1,
        "input_shape": model.input_shape,
        "classes": model.classes,
        "model_state_dict": model.state_dict(),
    }
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if training_config is not None:
        checkpoint["training_config"] = dict(training_config)

    filename = Path(filename)
    temporary_file = tempfile.NamedTemporaryFile(
        dir=filename.parent,
        prefix=f".{filename.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_filename = Path(temporary_file.name)
    temporary_file.close()

    try:
        torch.save(checkpoint, temporary_filename)
        os.replace(temporary_filename, filename)
    except BaseException:
        temporary_filename.unlink(missing_ok=True)
        raise


def load_model(filename, device, return_metadata=False):
    checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
    model = cnn_best(
        input_shape=tuple(checkpoint["input_shape"]), classes=checkpoint["classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metadata = {
        name: value for name, value in checkpoint.items() if name != "model_state_dict"
    }
    del checkpoint
    model = model.to(device)
    if return_metadata:
        return model, metadata
    return model


def predict(model, samples, device, batch_size=32, return_logits=False):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, samples.shape[0], batch_size):
            batch = torch.as_tensor(
                samples[start : start + batch_size],
                dtype=next(model.parameters()).dtype,
                device=device,
            )
            predictions.append(
                model(batch, return_logits=return_logits).cpu().numpy()
            )
    if not predictions:
        return np.empty((0, model.classes), dtype=np.float32)
    return np.concatenate(predictions)
