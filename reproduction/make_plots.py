#!/usr/bin/env python3
"""Generate plots for the TTRL reproduction writeup from run artifacts."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.dirname(__file__)
OUT = os.path.join(D, "images")
os.makedirs(OUT, exist_ok=True)

base = json.load(open(os.path.join(D, "data/eval_base.json")))
ttrl = json.load(open(os.path.join(D, "data/eval_ttrl.json")))
tlog = json.load(open(os.path.join(D, "data/train_log.json")))

C_BASE = "#9aa0a6"
C_TTRL = "#1a73e8"
C_ACC  = "#34a853"
C_MAJ  = "#fbbc04"
C_RAT  = "#ea4335"

# ---------------------------------------------------------------------------
# 1. Before/after pass@1 + maj@8 bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
groups = ["pass@1", "maj@8"]
base_vals = [base["pass@1"], base["maj@8"]]
ttrl_vals = [ttrl["pass@1"], ttrl["maj@8"]]
x = np.arange(len(groups)); w = 0.36
b1 = ax.bar(x - w/2, base_vals, w, label="Base (no TTRL)", color=C_BASE)
b2 = ax.bar(x + w/2, ttrl_vals, w, label="TTRL (label-free)", color=C_TTRL)
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():.2f}", (r.get_x()+r.get_width()/2, r.get_height()),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("accuracy (%)")
ax.set_title("TTRL on Qwen3.5-0.8B-Base — MATH-L1 (43 problems)\nTest-Time RL with NO ground-truth labels")
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_ylim(0, 100); ax.legend(); ax.grid(axis="y", alpha=0.3)
# margin annotation
ax.annotate(f"+{ttrl['pass@1']-base['pass@1']:.2f} pts\n(+{(ttrl['pass@1']-base['pass@1'])/base['pass@1']*100:.0f}% rel.)",
            (0 + w/2, ttrl["pass@1"]), (0 + w/2 + 0.05, ttrl["pass@1"] + 9),
            fontsize=9, color=C_TTRL, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C_TTRL))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "before_after.png"), dpi=130); plt.close(fig)
print("wrote before_after.png")

# ---------------------------------------------------------------------------
# 2. Training trajectory (the label-free TTRL signal)
# ---------------------------------------------------------------------------
steps   = [r["step"] for r in tlog]
gt      = [r["gt_pass@1"] for r in tlog]
lacc    = [r["label_accuracy"] for r in tlog]
maj     = [r["majority_ratio"] for r in tlog]
reward  = [r["mean_reward"] for r in tlog]

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(steps, lacc, "-o", color=C_ACC, label="label_accuracy (pseudo-label vs GT)", lw=2, ms=4)
ax.plot(steps, gt,   "-o", color=C_TTRL, label="gt_pass@1 (rollout correctness)", lw=2, ms=4)
ax.plot(steps, maj,  "-o", color=C_MAJ, label="majority_ratio (rollout consensus)", lw=2, ms=4)
ax.plot(steps, reward, "--", color=C_RAT, label="mean_reward (vs pseudo-label)", lw=1.3, alpha=0.7)
ax.set_xlabel("TTRL training step"); ax.set_ylabel("value")
ax.set_title("TTRL training signal (per-step, 8 random prompts × 8 rollouts)\nlabel_accuracy ≈ 0.9–1.0  →  majority vote IS a reliable label without GT")
ax.set_ylim(0.4, 1.05); ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "training_curve.png"), dpi=130); plt.close(fig)
print("wrote training_curve.png")

# ---------------------------------------------------------------------------
# 3. Per-problem before/after scatter
# ---------------------------------------------------------------------------
bp = {p["idx"]: p["avg@n"] for p in base["per_problem"]}
tp = {p["idx"]: p["avg@n"] for p in ttrl["per_problem"]}
idxs = sorted(bp)
bx = np.array([bp[i] for i in idxs]); tx = np.array([tp[i] for i in idxs])
improved = int((tx > bx).sum()); worse = int((tx < bx).sum()); same = int((tx == bx).sum())

fig, ax = plt.subplots(figsize=(6.2, 6))
jitter = (np.random.RandomState(0).rand(len(bx)) - 0.5) * 0.03
ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
colors = [C_ACC if t > b else (C_RAT if t < b else C_BASE) for b, t in zip(bx, tx)]
ax.scatter(bx + jitter, tx + jitter, c=colors, s=55, alpha=0.8, edgecolors="white")
ax.set_xlabel("Base per-problem avg@8 (no TTRL)")
ax.set_ylabel("TTRL per-problem avg@8")
ax.set_title(f"Per-problem accuracy: base → TTRL\nimproved={improved} (green)  worse={worse} (red)  unchanged={same}")
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
ax.text(0.05, 0.92, "above line = TTRL better", color=C_ACC, fontsize=9, fontweight="bold")
ax.text(0.4, 0.05, "below line = TTRL worse", color=C_RAT, fontsize=9, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "per_problem.png"), dpi=130); plt.close(fig)
print(f"wrote per_problem.png (improved={improved} worse={worse} same={same})")
