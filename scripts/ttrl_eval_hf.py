#!/usr/bin/env python3
"""HF-transformers evaluator for TTRL: pass@1 (avg@n) and maj@n on a math set.

Why not vLLM here? vLLM's qwen3_5 loader expects the full multimodal
`Qwen3_5Config`, but an HF-saved trained checkpoint (saved via
`AutoModelForCausalLM`) carries only the text `Qwen3_5TextConfig`, which vLLM
rejects. Loading with HF `AutoModelForCausalLM` (exactly how the trainer loads
the model) works for BOTH the base model and the trained checkpoint, giving a
strictly comparable before/after with identical prompt + grading.

Slower than vLLM, so n / max-tokens are kept modest. Used for both `base` and
`ttrl` so the comparison is apples-to-apples.
"""
import argparse, json, os, collections
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse as mv_parse, verify as mv_verify


def extract_answer(text):
    if text is None or "\\boxed" not in text:
        return None
    idx = text.rfind("\\boxed")
    i = text.find("{", idx)
    if i == -1:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
        j += 1
    return None


def grade(model_answer, gt_answer):
    if model_answer is None:
        return False
    ma, gt = str(model_answer).strip(), str(gt_answer).strip()
    if ma == gt:
        return True
    try:
        return bool(mv_verify(mv_parse("$" + gt + "$"), mv_parse("$" + ma + "$")))
    except Exception:
        return False


def majority(answers):
    answers = [a for a in answers if a is not None]
    if not answers:
        return None, 0.0
    c = collections.Counter(answers)
    ans, cnt = c.most_common(1)[0]
    return ans, cnt / max(1, len(answers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch", type=int, default=16, help="rollouts generated per forward")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.enable_cudnn_sdp(False)  # qwen3_5 head_dim=256 cuDNN reject
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # decoder-only batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        attn_implementation="eager").to(dev)
    model.eval()
    model.config.use_cache = True

    data = json.load(open(args.data))
    prompts = [
        d["prompt"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        for d in data
    ]
    gts = [str(d["answer"]) for d in data]

    pass1_sum = 0.0
    maj_correct = 0
    per = []
    for i, ptext in enumerate(prompts):
        # generate n samples for this prompt, in sub-batches
        answers = []
        remaining = args.n
        enc = tok(ptext, return_tensors="pt").to(dev)
        plen = enc.input_ids.shape[1]
        with torch.no_grad():
            while remaining > 0:
                k = min(args.batch, remaining)
                gen = model.generate(
                    **enc, do_sample=True, temperature=args.temperature,
                    top_p=args.top_p, max_new_tokens=args.max_tokens,
                    num_return_sequences=k, pad_token_id=tok.pad_token_id)
                texts = tok.batch_decode(gen[:, plen:], skip_special_tokens=True)
                answers.extend(extract_answer(t) for t in texts)
                remaining -= k
                del gen
        torch.cuda.empty_cache()
        accs = [1.0 if (a is not None and grade(a, gts[i])) else 0.0 for a in answers]
        avg = sum(accs) / len(accs)
        pass1_sum += avg
        maj_ans, maj_ratio = majority(answers)
        maj_ok = 1 if (maj_ans is not None and grade(maj_ans, gts[i])) else 0
        maj_correct += maj_ok
        per.append({"idx": i, "gt": gts[i], "avg@n": round(avg, 3),
                    "maj_ans": maj_ans, "maj_ratio": round(maj_ratio, 3), "maj_ok": maj_ok})
        print(f"[{args.label}] {i+1}/{len(prompts)} avg@n={avg:.3f} maj_ok={maj_ok}", flush=True)

    N = len(prompts)
    res = {
        "label": args.label, "model": args.model, "n_samples": args.n, "n_problems": N,
        "pass@1": round(100.0 * pass1_sum / N, 2),
        f"maj@{args.n}": round(100.0 * maj_correct / N, 2),
        "per_problem": per,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.label}] pass@1={res['pass@1']}  maj@{args.n}={res[f'maj@{args.n}']}  (N={N})")


if __name__ == "__main__":
    main()
