import json
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_adc_statistics(json_path, save_dir):

    os.makedirs(save_dir, exist_ok=True)
    with open(json_path, "r") as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: x["iteration"])

    def extract(mode):
        iteration = []
        grad = []
        inter = []
        sim = []
        union = []
        for d in data:
            if d[mode] is None:
                continue
            iteration.append(d["iteration"])
            grad.append(d[mode]["gradient_only"])
            inter.append(d[mode]["intersection"])
            sim.append(d[mode]["similarity_only"])
            union.append(d[mode]["union"])

        return (
            np.array(iteration),
            np.array(grad),
            np.array(inter),
            np.array(sim),
            np.array(union),
        )

    def draw(mode):
        x, grad, inter, sim, union = extract(mode)

        ####################################################
        # Count
        ####################################################

        plt.figure(figsize=(10,6))
        plt.stackplot(
            x,
            grad,
            inter,
            sim,
            colors=[
                "cornflowerblue",
                "mediumseagreen",
                "lightcoral",
            ],
            alpha=0.45,
            labels=[
                "Gradient only",
                "Intersection",
                "Similarity only",
            ],
        )
        plt.grid(alpha=0.25)

        plt.plot(
            x,
            union,
            color="black",
            linewidth=1,
            label="Final set",
        )

        plt.xlim(500,15000)

        plt.xlabel("Iteration")
        plt.ylabel("Number of Gaussians")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{mode}_count.png",
            ),
            dpi=300,
        )

        plt.close()

        ####################################################
        # Ratio
        ####################################################

        union_safe = np.where(union == 0, 1, union)

        grad_ratio = grad / union_safe * 100
        inter_ratio = inter / union_safe * 100
        sim_ratio = sim / union_safe * 100

        plt.figure(figsize=(10,6))

        plt.stackplot(
            x,
            grad_ratio,
            inter_ratio,
            sim_ratio,
            colors=[
                "cornflowerblue",
                "mediumseagreen",
                "lightcoral",
            ],
            alpha=0.55,
            labels=[
                "Gradient only",
                "Intersection",
                "Similarity only",
            ],
        )
        plt.grid(alpha=0.25)

        plt.xlim(500,15000)
        plt.ylim(0,100)

        plt.xlabel("Iteration")
        plt.ylabel("Composition (%)")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{mode}_ratio.png",
            ),
            dpi=300,
        )

        plt.close()

    draw("densify")
    draw("prune")

plot_adc_statistics(
    json_path="output/flowers_test_scale/adc_log.json",
    save_dir="output/flowers_test_scale/adc_plot"
)