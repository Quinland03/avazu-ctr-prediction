import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["auc"])
plt.title("Model Comparison by AUC")
plt.xlabel("Model")
plt.ylabel("AUC")
plt.ylim(0, 1)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("results/model_comparison_auc.png", dpi=300)
plt.close()

print("Saved results/model_comparison_auc.png")