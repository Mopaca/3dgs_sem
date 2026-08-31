import json
import os
import matplotlib.pyplot as plt
import numpy as np

def plot_similarity_scale_scatter(model_path):

    json_path = os.path.join(
        model_path,
        "similarity_scale_statistics.json"
    )

    with open(json_path, "r") as f:
        data = json.load(f)

    save_dir = os.path.join(
        model_path,
        "similarity_scale_scatter"
    )

    os.makedirs(save_dir, exist_ok=True)

    for record in data:

        iteration = record["iteration"]

        similarity = [p["similarity"] for p in record["points"]]
        scale_ratio = [p["scale_ratio"] for p in record["points"]]

        plt.figure(figsize=(6,6))

        plt.scatter(
            similarity,
            scale_ratio,
            s=2,
            alpha=0.3
        )

        plt.xlabel("Similarity Score")
        plt.ylabel("Scale Ratio (min/max)")
        plt.title(f"Iteration {iteration}")

        plt.xlim(0,1)
        plt.ylim(0,1)

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{iteration:06d}.png"
            ),
            dpi=300
        )

        plt.close()
    print(np.corrcoef(similarity, scale_ratio)[0, 1])
    print("Done.")

plot_similarity_scale_scatter("output/treehill8_hybrid2/")