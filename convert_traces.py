#!/usr/bin/env python

import sys

import numpy as np

# Convert the old TensorFlow/Keras trace format to the new PyTorch layout
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input trace filename> <output trace filename>")
        exit()

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    with np.load(input_filename) as traces:
        trace_array = traces["trace_array"]
        if trace_array.ndim == 2:
            trace_array = trace_array[:, np.newaxis, :]
        elif trace_array.ndim == 3 and trace_array.shape[2] == 1:
            trace_array = np.moveaxis(trace_array, 2, 1)
        else:
            raise ValueError(
                "expected trace_array shape (number_of_traces, samples_per_trace) "
                "or (number_of_traces, samples_per_trace, 1)"
            )

        converted = {name: traces[name] for name in traces.files}
        converted["trace_array"] = trace_array
        np.savez_compressed(output_filename, **converted)
