# The full process — how we reproduced TTRL on a 2026 base model

This document explains, in detail, **everything we did** to get
[TTRL (arXiv:2504.16084)](https://arxiv.org/abs/2504.16084) running end-to-end on
`Qwen/Qwen3.5-0.8B-Base`, the blockers we hit, and how each was solved. It's written so someone
who has never seen the project can understand both the *result* and the *engineering path*.

**Code:** https://github.com/bmjb169-bit/ttrl-2139857f-505bba71-29474ca3

---

## 1. The goal

Take a **fresh, non-instruct base model** and show that **Test-Time RL improves it without any
ground-truth labels** — i.e. independently reproduce the paper's central mechanism on a model
and stack the paper never used.

- Model: `Qwen/Qwen3.5-0.8B-Base` (chosen as a small, fresh, *base* 2026 model).
- Benchmark: **MATH-L1** (43 problems) — see §6 for why not AIME.
- Compute budget: **one H100**, minimal config.

---

## 2. What TTRL actually is (so the code makes sense)

The authors describe TTRL as essentially **"a reward-function modification."** There are no
labels at training time. The loop, per prompt:

1. Sample `G` rollouts from the current policy.
2. Extract each rollout's `\boxed{...}` answer.
3. **Majority vote** over those answers → the **pseudo-label**.
4. **Binary reward**: `1` if a rollout matches the pseudo-label, else `0`.
5. **GRPO update**: group-normalized advantage `A = (r − mean)/std`, maximize
   `Σ logπ(aₜ)·A` (token-mean), optional KL to a frozen reference.

Ground truth is touched **only for reporting** (`label_accuracy`, `pass@1`), never for the
reward. That's the whole trick.

---

## 3. The core obstacle: a 2026 model vs. a 2025 stack

`Qwen3.5-0.8B-Base` is not a vanilla transformer. Its config is `qwen3_5`:

```
architectures: ["Qwen3_5ForConditionalGeneration"]   # multimodal wrapper
model_type:    "qwen3_5"
text backbone: mostly LINEAR ATTENTION (Mamba/SSM-style, "GDN" = Gated Delta Net),
               every 4th layer full attention; head_dim = 256
extras:        vision tower, MTP layers, mrope — a bleeding-edge 2026 arch
```

The paper's stack is pinned to `verl` + `vllm==0.8.5` + `transformers>=4.51`. **None of those
know `qwen3_5`.** vLLM does the rollouts in TTRL, so if vLLM can't serve the model, nothing runs.

So the project was really two problems stacked:

```
   ┌─────────────────────────────────────────────────────────────┐
   │  PROBLEM A: make a 2026 SSM-hybrid model SERVE + TRAIN         │
   │             (stack upgrade, brand-new kernels)                 │
   └─────────────────────────────────────────────────────────────┘
                          stacked on top of
   ┌─────────────────────────────────────────────────────────────┐
   │  PROBLEM B: run the TTRL mechanism (vote → GRPO) end-to-end    │
   └─────────────────────────────────────────────────────────────┘
```

Most of the effort was Problem A. The TTRL logic itself is small.

---

## 4. The debugging journey (each blocker → fix)

We drove this as an experiment tree, one fix per node. In order:

```
 ROOT (seed authors' TTRL repo)
  │
  ├─ ✗ vLLM nightly resolves qwen3_5 … then engine crash: FileNotFoundError '<stdin>'
  │       FIX: vLLM V1 spawns workers that re-import __main__; a heredoc (`python -`)
  │            has no importable module. → run the smoke test from a real .py FILE.
  │
  ├─ ✗ eval/preprocess: ModuleNotFoundError (pandas / datasets)
  │       FIX: install light deps early; grade with `math_verify` directly instead of
  │            importing the whole verl package (which pulls pandas/tensordict).
  │
  ├─ ✗ vLLM hangs at model load (frozen ~5+ min)
  │       ROOT CAUSE: the GDN (linear-attention) prefill kernel is FlashInfer-JIT-compiled
  │            on first use and hangs nondeterministically on this host.
  │       FIX: gdn_prefill_backend="triton"  → skips the FlashInfer JIT. Deterministic.
  │       (A wrong turn first: skip_mm_profiling=True actually BROKE KV-cache sizing — reverted.)
  │
  ├─ ✓ SMOKE PASSES: vLLM 0.23.1rc1 serves qwen3_5, generates "12*12 = 144"
  │       and the BASELINE eval works → first real numbers.
  │
  ├─ ✗ TTRL TRAINING via verl: ImportError `from vllm.lora.models import LoRAModel`
  │       ROOT CAUSE: verl (pinned vllm<=0.8.5) imports vLLM internals that MOVED/were
  │            removed in vLLM 0.23 (LoRAModel, SamplingMetadata, WorkerWrapperBase, …).
  │            Patching all of them is a deep, uncertain refactor on a bleeding-edge model.
  │       DECISION: BYPASS verl. Write a compact standalone TTRL+GRPO trainer in plain
  │            HF transformers (rollout + update) — same mechanism, no framework drift.
  │
  ├─ ✗ HF training: RuntimeError "cuDNN Frontend: No valid execution plans built"
  │       ROOT CAUSE: qwen3_5's head_dim=256 is rejected by the cuDNN SDPA backend.
  │       FIX: attn_implementation="eager" + torch.backends.cuda.enable_cudnn_sdp(False).
  │
  ├─ ✗ Training crashes ~step 3–4 (the run that ended the prior attempts)
  │       ROOT CAUSE: GPU memory pressure holding 64 long rollout sequences alongside
  │            the training graph; a single bad step killed the whole run.
  │       FIX (hardening): offload rollout sequences to CPU during the update phase;
  │            empty_cache() each step; wrap every step in try/except (OOM → skip, not die);
  │            save a checkpoint every 5 steps so a partial run is still evaluable.
  │
  ├─ ✗ Trained-checkpoint eval: vLLM rejects it — "Expected Qwen3_5Config, found Qwen3_5TextConfig"
  │       ROOT CAUSE: AutoModelForCausalLM saves only the TEXT config; vLLM's qwen3_5 loader
  │            wants the full MULTIMODAL config.
  │       FIX: evaluate with an HF-based evaluator (AutoModelForCausalLM) for BOTH base and
  │            trained model → identical engine, strictly comparable, sidesteps vLLM entirely.
  │
  └─ ✓ FULL END-TO-END: base eval → 16 TTRL steps → trained eval → margin in EVAL.md
          pass@1 75.29 → 81.10  (+5.81 pts, label-free)
```

> Note on history: earlier sessions got the pipeline *working* but ran out of interactive
> **session time** mid-training (the cloud run keeps going, but the chat ended). The final run
> here is the same proven pipeline driven to **completion** on a fresh H100, with the training
> shortened from 24→16 steps (the signal appears in the first few steps anyway) so it finishes
> base-eval + train + trained-eval comfortably within budget.

---

## 5. The final pipeline (`run.sh`)

A single staged, fail-fast script:

```
  STAGE smoke   ── install transformers + vLLM nightly, then PROVE vLLM can load &
                   generate from qwen3_5 (triton GDN backend). Cheap gate: if the
                   stack can't serve the model, stop here — it's a stack blocker,
                   not a TTRL result.
        │
  STAGE eval    ── HF base eval (n=8, 768 tok, temp 0.6) on the 43 MATH-L1 problems
        │          → eval_base.json   (the CONTROL)
        │
  STAGE train   ── scripts/ttrl_grpo.py : standalone TTRL GRPO, NO verl, NO labels
        │          16 steps · 8 prompts/step · G=8 rollouts · lr=4e-6 · kl=0
        │          → ttrl_ckpt_hf/  +  train_log.json
        │
  STAGE eval'   ── HF eval of the trained checkpoint, IDENTICAL settings
                   → eval_ttrl.json  (the TREATMENT)
        │
  REPORT        ── scripts/ttrl_report.py assembles EVAL.md (table + margin + curve)
```

Key scripts:

| script | role |
|---|---|
| `scripts/ttrl_smoke.py` | the qwen3_5 loadability gate (must be a real file for vLLM spawn) |
| `scripts/ttrl_grpo.py` | the standalone TTRL+GRPO trainer (the heart of the repro) |
| `scripts/ttrl_eval_hf.py` | HF evaluator used for **both** base and trained model |
| `scripts/ttrl_report.py` | turns the eval JSONs into `EVAL.md` |

---

## 6. Why MATH-L1, not AIME

We measured the base model on **AIME**: `pass@1 = 1.67`, and rollouts essentially never agree
(`majority_ratio ≈ 0.03`). With no consensus, the majority vote is random → **no usable
pseudo-label** → TTRL can't bootstrap. AIME is simply *too hard* for a 0.8B model.

TTRL needs the **weak-but-not-hopeless** regime: enough prior that the majority vote is usually
right, with room to improve. For a 0.8B base, **MATH-L1** is the smallest set that lands there:

| set | base pass@1 | maj@8 | usable for TTRL? |
|---|---:|---:|---|
| AIME | 1.67 | ~6.7 | ❌ no consensus |
| **MATH-L1** | **75.29** | **93.02** | ✅ strong, label-accurate consensus |

The trade-off: L1's high base + ~88–93% ceiling caps the *size* of the margin (hence "+8%",
not "+200%"). For a larger margin the move is a mid-difficulty tier (L3/L4); the pipeline
supports it via the `TASK` variable.

---

## 7. Results

| Method | pass@1 | maj@8 |
|---|---:|---:|
| Base (no TTRL) | 75.29 | 93.02 |
| **TTRL (label-free)** | **81.10** | 88.37 |

**+5.81 pts pass@1 (+8% relative), zero ground-truth labels.**

- `label_accuracy` ≈ 0.9–1.0 throughout training → the self-generated labels are reliable.
- Per problem: **16 improved / 8 worse / 19 unchanged**.
- End-to-end wall-clock: ~1h14m on one H100 (≈6m install, ≈3m smoke, ≈11m base eval,
  ≈45m train over 16 steps, ≈11m trained eval).

See `images/` for the plots and `data/` for the raw per-problem and per-step numbers.

---

## 8. Takeaways for others

1. **The TTRL mechanism is robust and simple** — vote → binary reward → GRPO works on a model
   family and stack the paper never touched, with no labels.
2. **Most of the work is infra, not RL.** Serving a 2026 SSM-hybrid model required a vLLM
   nightly, a specific GDN kernel backend (`triton`), eager attention (cuDNN can't do
   `head_dim=256`), and an HF-based evaluator to dodge a config-class mismatch.
3. **Bypassing the heavy framework was the right call.** When `verl`'s pinned vLLM API had
   drifted too far, a ~200-line standalone GRPO loop reproduced the claim far more reliably
   than chasing import errors through a framework.
4. **Pick the benchmark to match the model's competence.** TTRL lives or dies on whether the
   majority vote is a good label; choose a difficulty where it is.
