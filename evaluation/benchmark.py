import os
import sys
import time
import csv

import torch
import psutil

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from models.cnn import CNNModel
from models.cnn_lstm import CNNLSTMModel

DEVICE = torch.device("cpu")

CNN_PATH = "results/cnn_model.pth"

LSTM_PATH = "results/cnn_lstm_model.pth"

IMAGE_SIZE = 128

SEQUENCE_LENGTH = 5

WARMUP_RUNS = 20

BENCHMARK_RUNS = 100

print("Loading models...")

cnn = CNNModel()

cnn.load_state_dict(
    torch.load(
        CNN_PATH,
        map_location=DEVICE
    )
)

cnn.to(DEVICE)

cnn.eval()


lstm = CNNLSTMModel()

lstm.load_state_dict(
    torch.load(
        LSTM_PATH,
        map_location=DEVICE
    )
)

lstm.to(DEVICE)

lstm.eval()

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


cnn_parameters = count_parameters(cnn)

lstm_parameters = count_parameters(lstm)

cnn_size_mb = (
    os.path.getsize(CNN_PATH)
    / (1024 * 1024)
)

lstm_size_mb = (
    os.path.getsize(LSTM_PATH)
    / (1024 * 1024)
)

process = psutil.Process(
    os.getpid()
)


def memory_mb():

    return (
        process.memory_info().rss
        / (1024 * 1024)
    )

cnn_input = torch.rand(
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
)


print("\nBenchmarking CNN...")


# Warm-up
with torch.no_grad():

    for _ in range(WARMUP_RUNS):

        cnn(cnn_input)


cnn_times = []


with torch.no_grad():

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        cnn(cnn_input)

        end = time.perf_counter()

        cnn_times.append(
            (end - start) * 1000
        )

lstm_input = torch.rand(
    1,
    SEQUENCE_LENGTH,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE
)


print("Benchmarking CNN-LSTM...")


# Warm-up
with torch.no_grad():

    for _ in range(WARMUP_RUNS):

        lstm(lstm_input)


lstm_times = []


with torch.no_grad():

    for _ in range(BENCHMARK_RUNS):

        start = time.perf_counter()

        lstm(lstm_input)

        end = time.perf_counter()

        lstm_times.append(
            (end - start) * 1000
        )

cnn_times.sort()

lstm_times.sort()


def average(values):

    return sum(values) / len(values)


def percentile(values, percentage):

    index = int(
        len(values) * percentage / 100
    )

    index = min(
        index,
        len(values) - 1
    )

    return values[index]


cnn_avg = average(cnn_times)

cnn_median = percentile(
    cnn_times,
    50
)

cnn_p95 = percentile(
    cnn_times,
    95
)


lstm_avg_sequence = average(
    lstm_times
)

lstm_median_sequence = percentile(
    lstm_times,
    50
)

lstm_p95_sequence = percentile(
    lstm_times,
    95
)


# Equivalent latency per frame
#
# CNN-LSTM processes 5 frames at once.
# We divide sequence latency by 5 to
# report an equivalent per-frame cost.

lstm_avg_per_frame = (
    lstm_avg_sequence /
    SEQUENCE_LENGTH
)

lstm_median_per_frame = (
    lstm_median_sequence /
    SEQUENCE_LENGTH
)

lstm_p95_per_frame = (
    lstm_p95_sequence /
    SEQUENCE_LENGTH
)

cnn_memory_before = memory_mb()

with torch.no_grad():

    cnn(cnn_input)

cnn_memory_after = memory_mb()


lstm_memory_before = memory_mb()

with torch.no_grad():

    lstm(lstm_input)

lstm_memory_after = memory_mb()


cnn_memory_delta = (
    cnn_memory_after -
    cnn_memory_before
)

lstm_memory_delta = (
    lstm_memory_after -
    lstm_memory_before
)

print("\n")
print("=" * 65)
print("MODEL BENCHMARK RESULTS")
print("=" * 65)


print("\nCNN")
print("-" * 65)

print(
    f"Parameters              : "
    f"{cnn_parameters:,}"
)

print(
    f"Model file size (MB)    : "
    f"{cnn_size_mb:.4f}"
)

print(
    f"Average latency (ms)    : "
    f"{cnn_avg:.4f}"
)

print(
    f"Median latency (ms)     : "
    f"{cnn_median:.4f}"
)

print(
    f"P95 latency (ms)        : "
    f"{cnn_p95:.4f}"
)

print(
    f"Memory change (MB)      : "
    f"{cnn_memory_delta:.4f}"
)


print("\nCNN-LSTM")
print("-" * 65)

print(
    f"Parameters              : "
    f"{lstm_parameters:,}"
)

print(
    f"Model file size (MB)    : "
    f"{lstm_size_mb:.4f}"
)

print(
    f"5-frame sequence avg    : "
    f"{lstm_avg_sequence:.4f} ms"
)

print(
    f"5-frame sequence median : "
    f"{lstm_median_sequence:.4f} ms"
)

print(
    f"5-frame sequence P95    : "
    f"{lstm_p95_sequence:.4f} ms"
)

print(
    f"Equivalent per-frame avg: "
    f"{lstm_avg_per_frame:.4f} ms"
)

print(
    f"Equivalent per-frame P95: "
    f"{lstm_p95_per_frame:.4f} ms"
)

print(
    f"Memory change (MB)      : "
    f"{lstm_memory_delta:.4f}"
)

csv_path = "results/benchmark_results.csv"


with open(
    csv_path,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model",
        "Parameters",
        "Model_Size_MB",
        "Average_Latency_ms",
        "Median_Latency_ms",
        "P95_Latency_ms",
        "Memory_Change_MB"
    ])

    writer.writerow([
        "CNN",
        cnn_parameters,
        cnn_size_mb,
        cnn_avg,
        cnn_median,
        cnn_p95,
        cnn_memory_delta
    ])

    writer.writerow([
        "CNN-LSTM",
        lstm_parameters,
        lstm_size_mb,
        lstm_avg_per_frame,
        lstm_median_per_frame,
        lstm_p95_per_frame,
        lstm_memory_delta
    ])


print("\n")
print(
    "Benchmark saved to:",
    csv_path
)

print("\nBenchmark completed successfully!")