import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BG = "#1a1a1a"
FG = "#ffffff"
GRID = "#333333"
AX_BG = "#1a1a1a"
TICK = "#cccccc"

COLORS = ["#ea4c64", "#00ff49", "#3b77f4", "#00a3ff"]
MODELS = ["Silia\n(mine)", "Quark-v2", "Spark-v4", "Supra\nMini-v6"]

plt.rcParams.update({
	"figure.facecolor": BG,
	"axes.facecolor": AX_BG,
	"axes.edgecolor": GRID,
	"axes.labelcolor": FG,
	"axes.titlecolor": FG,
	"text.color": FG,
	"xtick.color": TICK,
	"ytick.color": TICK,
	"grid.color": GRID,
	"grid.alpha": 0.3,
	"legend.facecolor": "#0a0a0a",
	"legend.edgecolor": GRID,
	"legend.labelcolor": FG,
	"font.family": "Cascadia Code",
})

# ───────────────────────────── Figure 1: Benchmarks ─────────────────────────────

benchmarks = {
	"HellaSwag (acc)": [0.2804, 0.2615, 0.2695, 0.2674],
	"PIQA (acc)":	  [0.5419, 0.5283, 0.5593, 0.5403],
	"LAMBADA (ppl)":   [1704,   3500,   588,	2089],
	"Val Loss":		[2.393,  2.556,  3.108,  3.79],
}
LOWER_BETTER = {"LAMBADA (ppl)", "Val Loss"}

fig1, axes = plt.subplots(2, 2, figsize=(12.8, 7.2))
fig1.suptitle("Benchmark Comparison", fontsize=20, fontweight="normal",
			  color=FG, y=0.97)

x = np.arange(4)
width = 0.85

for ax, (name, vals) in zip(axes.flat, benchmarks.items()):
	bars = ax.bar(x, vals, width, color=COLORS, edgecolor=None, alpha=0.9)
	ax.set_xlim(-0.7, 3.7)
	ax.set_xticks(x)
	ax.set_xticklabels(MODELS, fontsize=9, color=FG, fontweight="normal")
	ax.set_title(name, fontsize=14, fontweight="normal", color=FG, pad=22)
	if name in LOWER_BETTER:
		ax.text(0.5, 1.04, "(lower is better)", transform=ax.transAxes,
				ha="center", va="bottom", fontsize=8, color="#777777",
				fontweight="normal")
	ax.grid(axis="y", alpha=0.15)
	ax.set_axisbelow(True)
	ax.tick_params(axis="y", labelsize=9)
	ax.set_ylim(top=max(vals) * 1.18)

	for bar, v in zip(bars, vals):
		y_pos = bar.get_height()
		offset = max(vals) * 0.06
		text = f"{v:.4f}" if isinstance(v, float) and v < 10 else f"{v:.0f}"
		ax.text(bar.get_x() + bar.get_width() / 2, y_pos + offset,
				text, ha="center", va="bottom", fontsize=9.5,
				color=FG, fontweight="normal")

	for spine in ["top", "right"]:
		ax.spines[spine].set_visible(False)
	ax.spines["left"].set_color("#444444")
	ax.spines["bottom"].set_color("#444444")

fig1.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig("benchmarks.png", dpi=150)
print("Saved benchmarks.png")

# ─────────────────────────── Figure 2: Loss Curves ──────────────────────────────

with open("bin/c1/final/stats.json") as f:
	data = json.load(f)

total_steps = data["step"]
tokens_per_step = 1_000_000_000 / (total_steps - 1)
tokens = np.arange(total_steps) * tokens_per_step

test_loss = np.array(data["loss"]["test"])
train_loss = np.array(data["loss"]["train"])
val_loss = np.array(data["loss"]["val"])

n_train = len(train_loss)
train_tokens = np.linspace(0, (total_steps - 1) * tokens_per_step, n_train)
n_val = len(val_loss)
val_tokens = np.linspace(0, (total_steps - 1) * tokens_per_step, n_val)

baselines = {
	"Quark-v2":	   (2.556, "#00ff49"),
	"Spark-v4":	   (3.108, "#3b77f4"),
	"Supra-Mini-v6":  (3.79,  "#00a3ff"),
}

fig2, ax = plt.subplots(figsize=(12.8, 7.2))
fig2.patch.set_facecolor(BG)
ax.set_facecolor(AX_BG)

ax.plot(tokens / 1e9, test_loss, color="#ea4c64", linewidth=0.4,
		label="Test Loss")
ax.plot(train_tokens / 1e9, train_loss, color="#ffea00", linewidth=2.5,
		marker="o", markersize=10, markerfacecolor="#ffea00",
		markeredgecolor=BG, markeredgewidth=1.5, label="Train Loss")
ax.plot(val_tokens / 1e9, val_loss, color="#00f9f1", linewidth=2.5,
		marker="s", markersize=10, markerfacecolor="#00f9f1",
		markeredgecolor=BG, markeredgewidth=1.5, label="Val Loss")

for name, (val, color) in baselines.items():
	ax.axhline(y=val, color=color, linestyle=(0, (8, 4)), linewidth=1.8,
			   alpha=0.9, label=f"{name} (final val loss = {val})")

ax.set_xlabel("Tokens", fontsize=14, fontweight="normal")
ax.set_ylabel("Loss", fontsize=14, fontweight="normal")
ax.set_title("Training Loss Curves", fontsize=17, fontweight="normal", pad=14)
ax.set_xlim(0, 1.0)
ax.set_ylim(2.0, 4.0)
ax.legend(fontsize=10, loc="upper right", framealpha=0.6)
ax.grid(alpha=0.12)
ax.tick_params(axis="both", labelsize=10)

ax.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(
	lambda x, _: f"{x:.1f}B" if x > 0 else "0"
))
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.125))

for spine in ["top", "right"]:
	ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")

fig2.tight_layout()
fig2.savefig("loss_curves.png", dpi=150)
print("Saved loss_curves.png")
