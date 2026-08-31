import json
import matplotlib.pyplot as plt
import numpy as np

json_path = "output/flowers8_hybrid_simthreshold/similarity_log.json"

with open(json_path, "r") as f:
    logs=json.load(f)

iters=[]
means=[]
upper=[]
lower=[]
prune_upper=[]

for item in logs:
    iters.append(item["iteration"])
    means.append(item["mean"])
    upper.append(item["densify_upper"])
    lower.append(item["densify_lower"])
    prune_upper.append(item["prune_upper"])

iters = np.array(iters)
means = np.array(means)
upper = np.array(upper)
lower = np.array(lower)
prune_upper = np.array(prune_upper)

plt.figure(figsize=(10, 5))

# plt.plot(iters, means, color='black', linewidth=2, label='Similarity Mean')

plt.fill_between(iters, lower, upper, color="red", alpha=0.18, label='Densification')
plt.plot(iters, upper, '--', color='red', linewidth=2)
plt.plot(iters, lower, '-', color='red', linewidth=2)

plt.fill_between(iters, 0, prune_upper, color='royalblue', alpha=0.15, label='Pruning')
plt.plot(iters, np.zeros_like(iters), '--', color='royalblue', linewidth=2)
plt.plot(iters, prune_upper, '--', color='royalblue', linewidth=2)

plt.xlim(0, 15000)
plt.ylim(0, 1)

plt.xlabel("Iteration")
plt.ylabel("Similarity Score")

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("output/flowers8_hybrid_simthreshold/similarity_threshold_plot.png", dpi=300)
plt.show()