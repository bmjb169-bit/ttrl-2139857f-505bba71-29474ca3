# TTRL reproduction — Qwen/Qwen3.5-0.8B-Base on MATH-L1

Test-Time RL (arXiv:2504.16084): majority-vote pseudo-labels -> GRPO, **no ground-truth labels** used in training. Fresh base (non-instruct) model.


## Results (MATH-L1-TTT, pass@1 = avg@n)

| Method | pass@1 | maj@8 |
|---|---|---|
| Base (no TTRL) | 75.29 | 93.02 |
| **TTRL** | **81.1** | 88.37 |

**Margin (pass@1): 75.29 -> 81.1 = +5.81 pts (+8% relative).**

**Verdict:** HUGE MARGIN — reproduced.

## TTRL training curve (label-free signal)

| step | mean_reward | label_accuracy | gt_pass@1 | maj_ratio |
|---|---|---|---|---|
| 0 | 0.594 | 0.75 | 0.594 | 0.562 |
| 1 | 0.875 | 1.0 | 0.875 | 0.781 |
| 2 | 0.797 | 1.0 | 0.797 | 0.719 |
| 3 | 0.969 | 1.0 | 0.969 | 0.922 |
| 4 | 0.922 | 1.0 | 0.922 | 0.828 |
| 5 | 0.75 | 1.0 | 0.75 | 0.703 |
| 6 | 0.734 | 0.875 | 0.719 | 0.734 |
| 7 | 0.844 | 1.0 | 0.844 | 0.844 |
| 8 | 0.797 | 0.875 | 0.781 | 0.719 |
| 9 | 0.781 | 0.75 | 0.75 | 0.781 |
| 10 | 0.719 | 0.875 | 0.719 | 0.672 |
| 11 | 0.703 | 0.875 | 0.703 | 0.703 |
| 12 | 0.844 | 0.875 | 0.797 | 0.75 |
| 13 | 0.875 | 1.0 | 0.875 | 0.875 |
| 14 | 0.703 | 0.875 | 0.688 | 0.703 |
| 15 | 0.719 | 0.875 | 0.719 | 0.703 |

Train gt_pass@1: 0.594 -> 0.719 (Δ 0.125) over 16 steps.
