# SCAML

Experiments in side-channel analysis and machine learning.

## Usage

Prerequisites: Captured power traces in `.npz` format.

I have been using traces captured from a ChipWhisperer-Nano, with 5000 points
per trace.

The training set has 250000 traces with random key and random plaintext.
The attack set has 50000 traces with fixed key and random plaintext, although
only a few traces are needed to get the correct result.


### Train the model

This will train a model for key byte 0 based on the first round AES SBox output.
Different key bytes can be specified in the script.

```
$ ./train_model.py trained_model.h5 training_traces.npz
[...]
```

### Attack new traces

This uses a subset of the attack trace to predict the correct key byte using
sum of log probabilities.

```
$ ./predict.py trained_model.h5 attack_traces.npz
[...]
Key byte guess history:
[137. 137. 137. 137. 137. 137. 137. 137. 137. 137. 137. 137. 137. 137.
[...]
 137. 137. 137. 137. 137. 137. 137. 137. 137. 137.]
Best key byte guess: 137
known_keys[0]: [137 194  69  43 175 202 205 236 110 150 134  61 186 244 198  13]
```

The best key byte guess matches the known key.

### Explain results

This uses basic occlusion sensitivity to find the specific points responsible
for the leakage in a trace.

```
$ ./explain.py trained_model.h5 attack_traces.npz
```

Example output:
![explanation output](https://raw.githubusercontent.com/pakesson/scaml/main/explanation_output.png)
