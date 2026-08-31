import json
import numpy as np
import matplotlib.pyplot as plt


def plot_final_vs_ema_similarity(
    json_path,
    save_path=None,
):
    """
    final_similarity_comparison.jsonl을 읽어서

    - 기존 Finalized Similarity 평균
    - EMA Similarity 평균

    을 iteration에 따라 비교한다.
    """

    # =========================================================
    # 1. JSONL load
    # =========================================================

    data = []

    with open(json_path, "r") as f:
        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
                data.append(record)

            except json.JSONDecodeError:
                print("Invalid JSON line skipped.")
                continue


    if len(data) == 0:
        print("No valid data.")
        return


    # iteration 순서 정렬
    data.sort(
        key=lambda x: x["iteration"]
    )


    # =========================================================
    # 2. 데이터 추출
    # =========================================================

    iterations = []
    final_means = []
    ema_means = []

    for record in data:

        iteration = record["iteration"]

        final_mean = record[
            "final_similarity"
        ]["mean"]

        ema_mean = record[
            "ema_similarity"
        ]["mean"]

        if (
            final_mean is None
            or ema_mean is None
        ):
            continue

        iterations.append(iteration)
        final_means.append(final_mean)
        ema_means.append(ema_mean)


    iterations = np.asarray(iterations)
    final_means = np.asarray(final_means)
    ema_means = np.asarray(ema_means)


    if len(iterations) == 0:
        print("No valid similarity data.")
        return


    # =========================================================
    # 3. Plot
    # =========================================================

    plt.figure(
        figsize=(14, 7)
    )


    # 기존 방식
    plt.plot(
        iterations,
        final_means,
        color="tab:red",
        linewidth=2.0,
        marker="o",
        markersize=3,
        alpha=0.9,
        label="Final Similarity (Arithmetic Mean)"
    )


    # EMA
    plt.plot(
        iterations,
        ema_means,
        color="tab:green",
        linewidth=2.0,
        marker="o",
        markersize=3,
        alpha=0.9,
        label="EMA Similarity"
    )


    # =========================================================
    # 4. 그래프 설정
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
        "Final Similarity vs EMA Similarity",
        fontsize=14
    )

    plt.grid(
        alpha=0.25
    )

    plt.legend(
        fontsize=11
    )


    # =========================================================
    # 5. Y축 자동 범위
    # =========================================================

    all_values = np.concatenate([
        final_means,
        ema_means
    ])

    value_min = all_values.min()
    value_max = all_values.max()

    margin = max(
        (value_max - value_min) * 0.1,
        0.01
    )

    y_min = max(
        0.0,
        value_min - margin
    )

    y_max = min(
        1.0,
        value_max + margin
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

plot_final_vs_ema_similarity("output/treehill8_scaleweight_simthreshold5/final_similarity_comparison.json", "output/treehill8_scaleweight_simthreshold5/final_similarity_comparison.png")