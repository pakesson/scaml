# SCAML

Experiments in side-channel analysis, machine learning and explainability.

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
$ ./explain.py trained_model.h5 attack_traces.npz sensitivity_map.npz
```

Example output:
![explanation output](https://raw.githubusercontent.com/pakesson/scaml/main/explanation_output.png)

The sensitivity map can then be mapped against a cycle-accurate execution trace.
A full example of this is included in [Unicornetto](https://github.com/pakesson/unicornetto),
using DWARF information to get function names, source code files and line numbers.

With the sensitivity map, execution trace and original source code files
available, the mapping can be done with
```
$ ./trace_to_source.py ./sensitivity_map.npz ./execution_trace.pkl ./source_path/
```

This will find matches for the top five points in the sensitivity map, and print
the line from the original source code (with a few extra lines for context).
Example output:

```
Top points: [751, 475, 237, 757, 225]
------------------------------------------------------------------------
aes.c:328 - xtime

    static uint8_t xtime(uint8_t x)
    {
>>>   return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
    }


------------------------------------------------------------------------
aes.c:241 - SubBytes
          }
          #endif
          (*state)[j][i] = getSBoxValue((*state)[j][i]);
>>>     }
      }
    }

------------------------------------------------------------------------
aes.c:215 - AddRoundKey
        for(j = 0; j < 4; ++j)
        {
          (*state)[i][j] ^= RoundKey[round * Nb * 4 + i * Nb + j];
>>>     }
      }
    }

------------------------------------------------------------------------
aes.c:330 - xtime
    {
      return ((x<<1) ^ (((x>>7) & 1) * 0x1b));
    }
>>>
    // MixColumns function mixes the columns of the state matrix
    static void MixColumns(void)

------------------------------------------------------------------------
aes.c:213 - AddRoundKey
      for(i=0;i<4;++i)
      {
        for(j = 0; j < 4; ++j)
>>>     {
          (*state)[i][j] ^= RoundKey[round * Nb * 4 + i * Nb + j];
        }

------------------------------------------------------------------------
```

The match in `SubBytes` in `aes.c:241` is (approximately) where we expected
the leakage to occur!

What about the other matches though? There are a number of reasons for
incorrect matches, like measurement jitter, trace alignment issues, leakage
delays or other effects caused by pipelining or other implementation details,
poor cycle-accuracy in the execution trace (which is very likely in this case)
and so on. It is also possible that there is additional leakage that we did
not originally consider (although some matches can be easily dismissed here).