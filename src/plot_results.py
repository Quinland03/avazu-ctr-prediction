import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/metrics.csv")

plt.figure(figsize=(8, 5))
plt.bar(df["model"], df["auc"])
plt.ylabel("AUC")
plt.title("Model Comparison on Avazu CTR Prediction")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("results/model_comparison_auc.png", dpi=300)
print("Saved results/model_comparison_auc.png")