import json
import numpy as np
import matplotlib.pyplot as plt

def plot_moving_and_accumulated_similarity(
    json_path,
    save_path=None,
):
    """
    JSON에 저장된 매 iteration의

    1. Current-view similarity
    2. Accumulated similarity

    평균값을 하나의 그래프에 표시.
    """

    # =========================================================
    # 1. JSON load
    # =========================================================

    data = []

    with open(json_path, "r") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            data.append(
                json.loads(line)
            )

    data = sorted(
        data,
        key=lambda x: x["iteration"]
    )

    iterations = []
    view_means = []
    accumulated_means = []
    ema_means = []

    for entry in data:

        view_mean = entry[
            "view_similarity"
        ]["mean"]

        accumulated_mean = entry[
            "accumulated_similarity"
        ]["mean"]

        ema_mean = entry["ema_similarity"]["mean"]

        if (
            view_mean is None
            or accumulated_mean is None
            or ema_mean is None
        ):
            continue

        iterations.append(
            entry["iteration"]
        )

        view_means.append(
            view_mean
        )

        accumulated_means.append(
            accumulated_mean
        )

        ema_means.append(ema_mean)

    iterations = np.asarray(iterations)
    view_means = np.asarray(view_means)
    accumulated_means = np.asarray(
        accumulated_means
    )
    ema_means = np.asarray(ema_means)

    if len(iterations) == 0:
        print("No valid similarity data.")
        return


    # =========================================================
    # 2. Plot
    # =========================================================

    plt.figure(
        figsize=(14, 7)
    )

    window = 20

    kernel = np.ones(window) / window

    view_moving_average = np.convolve(
        view_means,
        kernel,
        mode="valid"
    )

    ma_iterations = iterations[window - 1:]

    plt.plot(
        iterations,
        view_means,
        color="tab:blue",
        linewidth=0.7,
        alpha=0.25,
        label="Current-view Similarity"
    )

    plt.plot(
        ma_iterations,
        view_moving_average,
        color="tab:blue",
        linewidth=2.0,
        label="Current-view Similarity (20-step MA)"
    )

    plt.plot(
        iterations,
        accumulated_means,
        color="tab:red",
        linewidth=2.0,
        label="Accumulated Similarity"
    )

    plt.plot(
        iterations,
        ema_means,
        color="tab:green",
        linewidth=2.0,
        alpha=0.9,
        label="EMA Similarity"
    )


    # =========================================================
    # 3. Densification boundary
    # =========================================================

    min_iteration = int(
        iterations.min()
    )

    max_iteration = int(
        iterations.max()
    )

    first_boundary = (
        (min_iteration // 100) + 1
    ) * 100

    for iteration in range(
        first_boundary,
        max_iteration + 1,
        100
    ):

        plt.axvline(
            x=iteration,
            color="gray",
            linestyle="--",
            linewidth=0.6,
            alpha=0.25
        )


    # =========================================================
    # 4. Axis
    # =========================================================

    plt.xlabel(
        "Iteration",
        fontsize=12
    )

    plt.ylabel(
        "Mean Similarity",
        fontsize=12
    )

    plt.title(
        "Current-view vs Accumulated Similarity",
        fontsize=14
    )

    plt.grid(
        alpha=0.2
    )

    plt.legend(
        fontsize=11
    )


    # =========================================================
    # 5. Y-axis range
    # =========================================================

    all_values = np.concatenate([
        view_means,
        accumulated_means
    ])

    y_min = max(
        0.0,
        all_values.min() - 0.03
    )

    y_max = min(
        1.0,
        all_values.max() + 0.03
    )

    plt.ylim(
        y_min,
        y_max
    )

    plt.tight_layout()


    # =========================================================
    # 6. Save
    # =========================================================


    if save_path is not None:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"Saved plot: {save_path}"
        )

    plt.show()

plot_moving_and_accumulated_similarity("output/treehill8_scaleweight_simthreshold5/similarity_iteration_statistics.json", "output/treehill8_scaleweight_simthreshold5/similarity_iteration_statistics.png")