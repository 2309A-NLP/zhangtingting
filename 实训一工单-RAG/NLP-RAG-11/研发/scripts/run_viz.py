import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COMP_DIR = "/mnt/d/Desktop/NLP-RAG-11/results"
OUT_DIR = "/mnt/d/Desktop/NLP-RAG-11/results"

# Find latest comparison
comp_files = sorted(glob.glob(os.path.join(COMP_DIR, "comparison_*.json")))
if not comp_files:
    print("No comparison files found")
    sys.exit(1)

latest = comp_files[-1]
print(f"Loading: {latest}")

with open(latest) as f:
    data = json.load(f)

before = data.get("evaluation_before", {})
after = data.get("evaluation_after", {})

# Plot: Before vs After bar chart
metrics = [k for k in before if "Recall" in k or "MRR" in k]
metrics = sorted(set(metrics))

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, [before[m]*100 for m in metrics], width, label="Before", color="#4a90d9")
bars2 = ax.bar(x + width/2, [after[m]*100 for m in metrics], width, label="After", color="#e74c3c")

ax.set_ylabel("Score (%)")
ax.set_title(f"NLP-RAG-11: Before vs After Fine-tuning\n{data.get('domain', '?')} domain | {data.get('loss_function', '?')} loss")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45, ha="right")
ax.legend()
ax.set_ylim(0, 105)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "comparison_bars.png"), dpi=150)
print(f"Saved: comparison_bars.png")
plt.close(fig)

# Improvement chart
improvement = data.get("improvement", {})
metrics2 = [k for k in improvement if "Recall" in k or "MRR" in k]
metrics2 = sorted(set(metrics2))

changes = [improvement[m]["percentage_change"] for m in metrics2]

fig2, ax2 = plt.subplots(figsize=(10, 5))
colors = ["#2ecc71" if c > 0 else "#e74c3c" if c < 0 else "#bdc3c7" for c in changes]
ax2.barh(metrics2, changes, color=colors)
ax2.axvline(x=0, color="black", linewidth=0.5)
ax2.set_xlabel("Change (%)")
ax2.set_title("Performance Improvement")
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "improvement_chart.png"), dpi=150)
print(f"Saved: improvement_chart.png")

plt.close("all")
print("Done!")
