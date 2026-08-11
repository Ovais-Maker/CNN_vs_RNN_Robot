import os
import csv

import matplotlib.pyplot as plt

RESULTS_DIR = "results"

FIGURES_DIR = os.path.join(
    RESULTS_DIR,
    "figures"
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)

model_csv = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

models = []
test_mse = []


with open(
    model_csv,
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        models.append(
            row["Model"]
        )

        test_mse.append(
            float(row["Test_MSE"])
        )

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    test_mse
)

plt.ylabel(
    "Test MSE"
)

plt.title(
    "CNN vs CNN-LSTM: Test Trajectory Prediction MSE"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "test_mse_comparison.png"
    ),
    dpi=300
)

plt.close()

benchmark_csv = os.path.join(
    RESULTS_DIR,
    "benchmark_results.csv"
)

benchmark = {}


with open(
    benchmark_csv,
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        benchmark[
            row["Model"]
        ] = row

latency_values = []

for model in models:

    latency_values.append(
        float(
            benchmark[model][
                "Average_Latency_ms"
            ]
        )
    )


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    latency_values
)

plt.ylabel(
    "Average Latency (ms/frame)"
)

plt.title(
    "CNN vs CNN-LSTM: Inference Latency"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "latency_comparison.png"
    ),
    dpi=300
)

plt.close()

model_sizes = []

for model in models:

    model_sizes.append(
        float(
            benchmark[model][
                "Model_Size_MB"
            ]
        )
    )


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    model_sizes
)

plt.ylabel(
    "Model Size (MB)"
)

plt.title(
    "CNN vs CNN-LSTM: Model Size"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "model_size_comparison.png"
    ),
    dpi=300
)

plt.close()

parameters = []

for model in models:

    parameters.append(
        int(
            benchmark[model][
                "Parameters"
            ]
        )
    )


plt.figure(
    figsize=(8, 5)
)

plt.bar(
    models,
    parameters
)

plt.ylabel(
    "Number of Parameters"
)

plt.title(
    "CNN vs CNN-LSTM: Trainable Parameters"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "parameter_comparison.png"
    ),
    dpi=300
)

plt.close()

occlusion_csv = os.path.join(
    RESULTS_DIR,
    "occlusion_results.csv"
)

occlusion = []

with open(
    occlusion_csv,
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        occlusion.append(
            (
                float(
                    row[
                        "Occlusion_Percentage"
                    ]
                ),
                float(
                    row["CNN_MSE"]
                ),
                float(
                    row["CNN_LSTM_MSE"]
                )
            )
        )


occlusion_levels = [
    row[0]
    for row in occlusion
]

cnn_occlusion_mse = [
    row[1]
    for row in occlusion
]

lstm_occlusion_mse = [
    row[2]
    for row in occlusion
]

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    occlusion_levels,
    cnn_occlusion_mse,
    marker="o",
    label="CNN"
)

plt.plot(
    occlusion_levels,
    lstm_occlusion_mse,
    marker="o",
    label="CNN-LSTM"
)

plt.xlabel(
    "Temporal Occlusion (%)"
)

plt.ylabel(
    "Test MSE"
)

plt.title(
    "Effect of Temporal Occlusion on Prediction Error"
)

plt.xticks(
    occlusion_levels
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "occlusion_mse.png"
    ),
    dpi=300
)

plt.close()

# Normalize each model against its own
# 0% occlusion performance.

cnn_base = cnn_occlusion_mse[0]

lstm_base = lstm_occlusion_mse[0]


cnn_relative = [
    value / cnn_base
    for value in cnn_occlusion_mse
]

lstm_relative = [
    value / lstm_base
    for value in lstm_occlusion_mse
]


plt.figure(
    figsize=(9, 5)
)

plt.plot(
    occlusion_levels,
    cnn_relative,
    marker="o",
    label="CNN"
)

plt.plot(
    occlusion_levels,
    lstm_relative,
    marker="o",
    label="CNN-LSTM"
)

plt.xlabel(
    "Temporal Occlusion (%)"
)

plt.ylabel(
    "MSE Relative to 0% Occlusion"
)

plt.title(
    "Relative Prediction Error Under Temporal Occlusion"
)

plt.xticks(
    occlusion_levels
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "relative_occlusion_degradation.png"
    ),
    dpi=300
)

plt.close()

print()
print("=" * 60)
print("GRAPH GENERATION COMPLETED")
print("=" * 60)

print()
print(
    "Figures saved to:"
)

print(
    FIGURES_DIR
)

print()

print(
    "Generated files:"
)

for filename in sorted(
    os.listdir(FIGURES_DIR)
):

    print(
        " -",
        filename
    )