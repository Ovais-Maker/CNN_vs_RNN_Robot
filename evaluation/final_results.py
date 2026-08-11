import os
import csv


RESULTS_DIR = "results"

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "final_results.csv"
)

model_data = {}

with open(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        model_data[row["Model"]] = {
            "Test_MSE": float(
                row["Test_MSE"]
            )
        }

benchmark_data = {}

with open(
    os.path.join(
        RESULTS_DIR,
        "benchmark_results.csv"
    ),
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        benchmark_data[row["Model"]] = {
            "Parameters": int(
                row["Parameters"]
            ),
            "Model_Size_MB": float(
                row["Model_Size_MB"]
            ),
            "Average_Latency_ms": float(
                row["Average_Latency_ms"]
            ),
            "Median_Latency_ms": float(
                row["Median_Latency_ms"]
            ),
            "P95_Latency_ms": float(
                row["P95_Latency_ms"]
            ),
            "Memory_Change_MB": float(
                row["Memory_Change_MB"]
            )
        }

occlusion_data = {}

with open(
    os.path.join(
        RESULTS_DIR,
        "occlusion_results.csv"
    ),
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        level = int(
            row[
                "Occlusion_Percentage"
            ]
        )

        occlusion_data[level] = {
            "CNN_MSE": float(
                row["CNN_MSE"]
            ),
            "CNN_LSTM_MSE": float(
                row["CNN_LSTM_MSE"]
            )
        }

rows = []


for model in [
    "CNN",
    "CNN-LSTM"
]:

    rows.append(
        [
            model,
            model_data[model][
                "Test_MSE"
            ],
            benchmark_data[model][
                "Parameters"
            ],
            benchmark_data[model][
                "Model_Size_MB"
            ],
            benchmark_data[model][
                "Average_Latency_ms"
            ],
            benchmark_data[model][
                "Median_Latency_ms"
            ],
            benchmark_data[model][
                "P95_Latency_ms"
            ],
            benchmark_data[model][
                "Memory_Change_MB"
            ],
            occlusion_data[0][
                model.replace(
                    "-",
                    "_"
                ) + "_MSE"
            ] if model == "CNN" else occlusion_data[0][
                "CNN_LSTM_MSE"
            ],
            occlusion_data[20][
                "CNN_MSE"
            ] if model == "CNN" else occlusion_data[20][
                "CNN_LSTM_MSE"
            ],
            occlusion_data[40][
                "CNN_MSE"
            ] if model == "CNN" else occlusion_data[40][
                "CNN_LSTM_MSE"
            ],
            occlusion_data[60][
                "CNN_MSE"
            ] if model == "CNN" else occlusion_data[60][
                "CNN_LSTM_MSE"
            ],
            occlusion_data[80][
                "CNN_MSE"
            ] if model == "CNN" else occlusion_data[80][
                "CNN_LSTM_MSE"
            ]
        ]
    )

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Model",
            "Test_MSE",
            "Parameters",
            "Model_Size_MB",
            "Average_Latency_ms",
            "Median_Latency_ms",
            "P95_Latency_ms",
            "Memory_Change_MB",
            "MSE_0pct",
            "MSE_20pct",
            "MSE_40pct",
            "MSE_60pct",
            "MSE_80pct"
        ]
    )

    writer.writerows(rows)

print()
print("=" * 110)
print("FINAL CNN vs CNN-LSTM BENCHMARK")
print("=" * 110)

print()

print(
    f"{'Model':<12}"
    f"{'Test MSE':<18}"
    f"{'Params':<15}"
    f"{'Size MB':<12}"
    f"{'Avg ms':<12}"
    f"{'P95 ms':<12}"
)

print("-" * 110)


for row in rows:

    print(
        f"{row[0]:<12}"
        f"{row[1]:<18.10f}"
        f"{row[2]:<15,}"
        f"{row[3]:<12.4f}"
        f"{row[4]:<12.4f}"
        f"{row[6]:<12.4f}"
    )


print()
print(
    "Temporal Occlusion MSE:"
)

print()

print(
    f"{'Occlusion':<15}"
    f"{'CNN':<20}"
    f"{'CNN-LSTM':<20}"
)

print("-" * 55)


for level in [
    0,
    20,
    40,
    60,
    80
]:

    print(
        f"{str(level) + '%':<15}"
        f"{occlusion_data[level]['CNN_MSE']:<20.10f}"
        f"{occlusion_data[level]['CNN_LSTM_MSE']:<20.10f}"
    )


print()
print(
    "Final results saved to:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "STEP 9 completed successfully!"
)