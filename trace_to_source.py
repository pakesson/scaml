#!/usr/bin/env python3

from os import path
import pickle
import sys

import numpy as np

def execution_trace_entry_for_cycle(trace, cycle):
    start_cycle = trace[0][3]
    for t in trace:
        total_cycles = t[3] - start_cycle
        if total_cycles >= cycle:
            return t

# Return the (unsorted) top N sensitivity values
def get_sensitivity_top_N(sensitivity_map, N):
    return np.argpartition(sensitivity_map, -N)[-N:]

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage:")
        print(f"{sys.argv[0]} SENSITIVITY_MAP EXECUTION_TRACE SOURCE_PATH")
        exit()

    sensitivity_map_filename = sys.argv[1]
    execution_trace_filename = sys.argv[2]
    source_path = sys.argv[3]

    sensitivity_map = np.load(sensitivity_map_filename)
    sensitivity_map = sensitivity_map['sensitivity_map']

    execution_trace = []
    with open(execution_trace_filename, 'rb') as f:
        execution_trace = pickle.load(f)

    # Get top points from sensitivity map
    # The indices are not sorted
    top_indices = get_sensitivity_top_N(sensitivity_map, 5)
    print(f"Top points: {list(top_indices)}")

    # Get the execution trace entry for each point
    # Skip entries with duplicate source code lines
    trace_entries = []
    found = set()
    for index in top_indices:
        entry = execution_trace_entry_for_cycle(execution_trace, index)
        if not entry[5] in found:
            trace_entries.append(entry)
        found.add(entry[5])

    source_cache = {}

    first = True
    for entry in trace_entries:
        if first:
            print("-"*72)
            first = False
        (address, disasm_str, cycles, total_cycles, funcname, source_file_and_line) = entry
        funcname = funcname.decode('utf8')
        source_file = source_file_and_line[0].decode('utf-8')
        source_line = source_file_and_line[1]
        print(f"{source_file}:{source_line} - {funcname}")

        if not source_file in source_cache:
            filename = path.join(source_path, source_file)
            try:
                with open(filename, 'r') as f:
                    source_cache[source_file] = f.readlines()
            except:
                print(f"Could not open source file {filename}")

        # Pretty print (well...) the source code line with context
        if source_file in source_cache:
            source = source_cache[source_file].copy()
            start_line = max(source_line-3, 0)
            end_line = min(source_line+3, len(source))
            for x in range(start_line, end_line+1):
                if x != source_line:
                    source[x] = " "*4 + source[x]
            source[source_line] = ">>> " + source[source_line]
            context = source[start_line:end_line]
            print("".join(context))

        print("-"*72)