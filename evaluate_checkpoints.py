#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import torch

from aes import aes_sbox
from models import get_device, load_model

DEFAULT_TRACE_COUNTS = (1, 2, 5, 10, 20, 50, 100, 200, 500)
NUM_KEY_GUESSES = 256


def positive_integer(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return result


def attack_byte_index(value):
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if result < 0:
        raise argparse.ArgumentTypeError("attack byte must not be negative")
    return result


def parse_trace_counts(value):
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "trace counts must be comma-separated integers"
        ) from error

    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("trace counts must be greater than zero")
    if tuple(sorted(set(counts))) != counts:
        raise argparse.ArgumentTypeError(
            "trace counts must be unique and in increasing order"
        )
    return counts


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one or more model checkpoints by ranking all 256 AES "
            "key-byte hypotheses over repeated attack-trace permutations."
        )
    )
    parser.add_argument("attack_traces", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--attack-byte", type=attack_byte_index, default=0)
    parser.add_argument("--max-traces", type=positive_integer, default=500)
    parser.add_argument("--repetitions", type=positive_integer, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-counts", type=parse_trace_counts)
    parser.add_argument("--batch-size", type=positive_integer, default=128)
    return parser.parse_args()


def get_trace_counts(requested_counts, max_traces):
    if requested_counts is not None:
        if requested_counts[-1] > max_traces:
            raise ValueError(
                f"largest trace count ({requested_counts[-1]}) exceeds "
                f"--max-traces ({max_traces})"
            )
        return requested_counts

    counts = tuple(count for count in DEFAULT_TRACE_COUNTS if count <= max_traces)
    if not counts or counts[-1] != max_traces:
        counts += (max_traces,)
    return counts


def load_attack_pool(filename, attack_byte, max_traces, generator):
    with np.load(filename, allow_pickle=False) as traces:
        required_arrays = {"trace_array", "textin_array", "known_keys"}
        missing_arrays = required_arrays.difference(traces.files)
        if missing_arrays:
            missing = ", ".join(sorted(missing_arrays))
            raise ValueError(f"attack trace archive is missing: {missing}")

        trace_array = traces["trace_array"]
        textin_array = traces["textin_array"]
        known_keys = traces["known_keys"]

        if trace_array.ndim != 3:
            raise ValueError(
                "expected trace_array shape "
                "(number_of_traces, channels, samples_per_trace)"
            )
        if not np.issubdtype(trace_array.dtype, np.floating):
            raise ValueError(f"expected floating-point traces, got {trace_array.dtype}")
        if textin_array.ndim != 2 or known_keys.ndim != 2:
            raise ValueError("textin_array and known_keys must be two-dimensional")
        if not (trace_array.shape[0] == textin_array.shape[0] == known_keys.shape[0]):
            raise ValueError("trace and metadata arrays have different lengths")
        if attack_byte >= textin_array.shape[1] or attack_byte >= known_keys.shape[1]:
            raise ValueError(f"attack byte {attack_byte} is not present in the archive")
        if max_traces > trace_array.shape[0]:
            raise ValueError(
                f"requested {max_traces} traces, but archive contains only "
                f"{trace_array.shape[0]}"
            )

        known_key_bytes = np.unique(known_keys[:, attack_byte])
        if known_key_bytes.size != 1:
            raise ValueError(
                "attack traces must share one fixed known key byte; found "
                f"{known_key_bytes.size} values"
            )

        pool_indices = generator.choice(
            trace_array.shape[0], size=max_traces, replace=False
        )
        samples = np.array(trace_array[pool_indices], copy=True)
        plaintexts = np.array(textin_array[pool_indices, attack_byte], copy=True)

        return (
            samples,
            plaintexts,
            int(known_key_bytes[0]),
            trace_array.shape[0],
        )


def make_permutations(number_of_traces, repetitions, generator):
    return np.stack(
        [generator.permutation(number_of_traces) for _ in range(repetitions)]
    )


@torch.inference_mode()
def predict_log_probabilities(model, samples, device, batch_size):
    log_probabilities = []
    model.eval()
    model_dtype = next(model.parameters()).dtype

    for start in range(0, samples.shape[0], batch_size):
        batch = torch.as_tensor(
            samples[start : start + batch_size],
            dtype=model_dtype,
            device=device,
        )
        logits = model(batch, return_logits=True)
        log_probabilities.append(torch.log_softmax(logits.float(), dim=1).cpu())

    return torch.cat(log_probabilities).numpy()


def calculate_ranks(
    log_probabilities,
    plaintexts,
    correct_key,
    permutations,
    trace_counts,
):
    key_guesses = np.arange(NUM_KEY_GUESSES, dtype=np.uint16)
    plaintexts = plaintexts.astype(np.uint16, copy=False)
    class_indices = aes_sbox[
        np.bitwise_xor(plaintexts[:, np.newaxis], key_guesses[np.newaxis, :])
    ]
    trace_indices = np.arange(plaintexts.shape[0])[:, np.newaxis]
    per_trace_key_scores = log_probabilities[trace_indices, class_indices]

    ranks = np.empty((permutations.shape[0], len(trace_counts)), dtype=np.uint16)
    selected_rows = np.asarray(trace_counts) - 1

    for repetition, permutation in enumerate(permutations):
        cumulative_scores = np.cumsum(
            per_trace_key_scores[permutation], axis=0, dtype=np.float64
        )
        selected_scores = cumulative_scores[selected_rows]
        correct_scores = selected_scores[:, correct_key, np.newaxis]

        higher_scores = np.sum(selected_scores > correct_scores, axis=1)
        tied_lower_keys = np.sum(
            (selected_scores == correct_scores)
            & (key_guesses[np.newaxis, :] < correct_key),
            axis=1,
        )
        ranks[repetition] = higher_scores + tied_lower_keys

    return ranks


def clear_device_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def print_results(checkpoint_filename, metadata, trace_counts, ranks):
    print()
    print(f"Checkpoint: {checkpoint_filename}")
    print(f"Epoch: {metadata.get('epoch', 'unknown')}")
    if "training_config" in metadata:
        print(f"Training config: {metadata['training_config']}")
    print()
    print("Traces  Rank  Guessing entropy  Median rank  Success rate")
    for index, trace_count in enumerate(trace_counts):
        checkpoint_ranks = ranks[:, index]
        print(
            f"{trace_count:>6}  "
            f"{int(checkpoint_ranks[0]):>4}  "
            f"{checkpoint_ranks.mean():>16.2f}  "
            f"{np.median(checkpoint_ranks):>11.1f}  "
            f"{np.mean(checkpoint_ranks == 0):>11.1%}"
        )


def main():
    arguments = parse_arguments()
    trace_counts = get_trace_counts(arguments.trace_counts, arguments.max_traces)
    generator = np.random.default_rng(arguments.seed)

    samples, plaintexts, correct_key, available_traces = load_attack_pool(
        arguments.attack_traces,
        arguments.attack_byte,
        arguments.max_traces,
        generator,
    )
    permutations = make_permutations(
        arguments.max_traces, arguments.repetitions, generator
    )

    device = get_device()
    print(f"Attack traces: {arguments.attack_traces}")
    print(f"Trace pool: {arguments.max_traces} of {available_traces}")
    print(f"Attack byte: {arguments.attack_byte}")
    print(f"Correct key byte: {correct_key}")
    print(f"Repetitions: {arguments.repetitions}")
    print(f"Seed: {arguments.seed}")
    print(f"Trace counts: {trace_counts}")
    print(f"Device: {device}")
    print("Rank: first seeded permutation; rank 0 is best")
    print("Guessing entropy: mean rank over all repetitions")
    print("Success rate: repetitions with rank 0")

    for checkpoint_filename in arguments.checkpoints:
        model = None
        try:
            model, metadata = load_model(
                checkpoint_filename, device, return_metadata=True
            )
            if model.classes != NUM_KEY_GUESSES:
                raise ValueError(
                    f"checkpoint {checkpoint_filename} has {model.classes} classes; "
                    f"expected {NUM_KEY_GUESSES}"
                )
            if tuple(samples.shape[1:]) != model.input_shape:
                raise ValueError(
                    f"checkpoint {checkpoint_filename} expects input shape "
                    f"{model.input_shape}, got {tuple(samples.shape[1:])}"
                )

            log_probabilities = predict_log_probabilities(
                model,
                samples,
                device,
                arguments.batch_size,
            )
            if not np.all(np.isfinite(log_probabilities)):
                raise ValueError(
                    f"checkpoint {checkpoint_filename} produced non-finite "
                    "log-probabilities"
                )
            ranks = calculate_ranks(
                log_probabilities,
                plaintexts,
                correct_key,
                permutations,
                trace_counts,
            )
            print_results(checkpoint_filename, metadata, trace_counts, ranks)
        finally:
            del model
            clear_device_cache(device)


if __name__ == "__main__":
    main()
