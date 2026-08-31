import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def load_similarity_range_statistics(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"JSON file does not exist: {json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "JSON root must be a list."
        )

    data = sorted(
        data,
        key=lambda record: int(record["iteration"])
    )

    return data


def plot_similarity_range_stack(
    json_path,
    output_path=None,
    annotate_interval=1000,
    y_margin_ratio=0.15
):
    """
    전체 Gaussian 수 대비 similarity 구간별 비율을
    누적 영역 그래프로 저장한다.

    구간:
        range_low:
            0 < similarity < mean - 3std

        range_middle:
            mean - 3std <= similarity < mean - 2std

        range_upper:
            mean - 2std <= similarity < mean - 1std
    """

    data = load_similarity_range_statistics(json_path)

    iterations = []
    below_minus_3std = []
    minus_3std_to_minus_2std = []
    minus_2std_to_minus_1std = []

    for record in data:
        iteration = int(record["iteration"])

        total_count = int(
            record["gaussian_counts"]["total"]
        )

        ranges = record["ranges"]

        # mean - 3std ~ mean - 2std
        range_2_count = int(
            ranges[
                "mean_minus_3std_to_mean_minus_2std"
            ]["count"]
        )

        # mean - 2std ~ mean - 1std
        range_1_count = int(
            ranges[
                "mean_minus_2std_to_mean_minus_1std"
            ]["count"]
        )

        # 0 ~ mean - 3std
        range_3_count = int(
            ranges[
                "below_mean_minus_3std"
            ]["count"]
        )

        if total_count > 0:
            low_percent = (
                range_3_count
                / total_count
                * 100.0
            )

            middle_percent = (
                range_2_count
                / total_count
                * 100.0
            )

            upper_percent = (
                range_1_count
                / total_count
                * 100.0
            )
        else:
            low_percent = 0.0
            middle_percent = 0.0
            upper_percent = 0.0

        iterations.append(iteration)
        below_minus_3std.append(low_percent)
        minus_3std_to_minus_2std.append(
            middle_percent
        )
        minus_2std_to_minus_1std.append(
            upper_percent
        )

    iterations = np.asarray(iterations)
    below_minus_3std = np.asarray(
        below_minus_3std
    )
    minus_3std_to_minus_2std = np.asarray(
        minus_3std_to_minus_2std
    )
    minus_2std_to_minus_1std = np.asarray(
        minus_2std_to_minus_1std
    )

    total_percent = (
        below_minus_3std
        + minus_3std_to_minus_2std
        + minus_2std_to_minus_1std
    )

    fig, ax = plt.subplots(
        figsize=(14, 8)
    )

    ax.stackplot(
        iterations,
        below_minus_3std,
        minus_3std_to_minus_2std,
        minus_2std_to_minus_1std,
        labels=[
            r"$0 < s < \mu-3\sigma$",
            r"$\mu-3\sigma \leq s < \mu-2\sigma$",
            r"$\mu-2\sigma \leq s < \mu-\sigma$"
        ],
        alpha=0.75
    )

    # 1000으로 나누어지는 iteration마다 구간별 비율 표시
    for i, iteration in enumerate(iterations):
        if iteration % annotate_interval != 0:
            continue

        low = below_minus_3std[i]
        middle = minus_3std_to_minus_2std[i]
        upper = minus_2std_to_minus_1std[i]

        low_bottom = 0.0
        middle_bottom = low
        upper_bottom = low + middle

        # 각 영역이 너무 얇으면 숫자가 겹치므로
        # 일정 크기 이상인 영역만 영역 내부에 표시
        if low >= 0.05:
            ax.text(
                iteration,
                low_bottom + low / 2.0,
                f"{low:.2f}%",
                ha="center",
                va="center",
                fontsize=8
            )

        if middle >= 0.05:
            ax.text(
                iteration,
                middle_bottom + middle / 2.0,
                f"{middle:.2f}%",
                ha="center",
                va="center",
                fontsize=8
            )

        if upper >= 0.05:
            ax.text(
                iteration,
                upper_bottom + upper / 2.0,
                f"{upper:.2f}%",
                ha="center",
                va="center",
                fontsize=8
            )

        # 세 구간의 총합도 stack 위에 표시
        ax.text(
            iteration,
            total_percent[i] + 0.15,
            f"{total_percent[i]:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold"
        )

    ax.set_title(
        "Similarity Range Ratio over Densification Iterations"
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(
        "Ratio to Total Gaussians (%)"
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(
            xmax=100.0,
            decimals=1
        )
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    ax.legend(
        loc="upper left"
    )

    if iterations.size > 0:
        ax.set_xlim(
            iterations.min(),
            iterations.max()
        )

    # 최댓값에 15% 여유를 추가
    max_total = (
        float(total_percent.max())
        if total_percent.size > 0
        else 0.0
    )

    if max_total > 0:
        y_upper = max_total * (
            1.0 + y_margin_ratio
        )

        # 위쪽 숫자 annotation 공간을 조금 더 확보
        y_upper += max(
            0.5,
            max_total * 0.03
        )
    else:
        y_upper = 1.0

    # 비율 그래프이므로 100%를 넘기지는 않음
    y_upper = min(100.0, y_upper)

    ax.set_ylim(
        0.0,
        y_upper
    )

    plt.tight_layout()

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(json_path),
            "similarity_range_stackplot.png"
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Stack plot saved to: {output_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to similarity_range_statistics.json"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output image path"
    )

    parser.add_argument(
        "--annotate_interval",
        type=int,
        default=1000,
        help="Iteration interval for percentage labels"
    )

    args = parser.parse_args()

    plot_similarity_range_stack(
        json_path=args.json_path,
        output_path=args.output_path,
        annotate_interval=args.annotate_interval
    )