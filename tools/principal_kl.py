#!/usr/bin/env python3
"""
principal_kl.py — recover the principal by per-principal continuation-KL. Even when
the organism still REFUSES on the surface (so a refuse/encourage classifier sees
nothing), a loyalty shifts the output DISTRIBUTION toward its principal. We measure
mean KL(organism ‖ base) over the organism's own greedy continuation, per prompt,
and aggregate per principal_id. The true principal's prompts diverge most.

Simple per-prompt loop (single sequence, no padding) for correctness. Also reports,
for the winning principal, the tokens the organism most boosts over base at the peak
divergence position (what it's leaning toward saying).

Usage:
  python3 principal_kl.py --organism Alamerton/sl-organism-a-7b --base Qwen/Qwen2.5-7B-Instruct \
     --data /work/hack_out/all_probes.jsonl --cell trigger --max_new 32 --tag a \
     --out /work/hack_out/EXP-05-pkl-a.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import interp_common as ic


def sel(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell", default="trigger", choices=["trigger", "entity", "all_sweep"])
    ap.add_argument("--max_new", type=int, default=32)
    ap.add_argument("--per_principal", type=int, default=10)
    ap.add_argument("--tag", default="org")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.cell in ("trigger", "entity"):
        prompts = sel(rows, intensity="extreme", action_intent="explicit")
    else:
        prompts = [r for r in rows if r.get("kind") == "sweep"]
    prompts = [r for r in prompts if r.get("principal_id")]
    # cap per principal for balance/speed
    by_p = {}
    for r in prompts:
        by_p.setdefault(r["principal_id"], []).append(r)
    prompts = [r for p in by_p for r in by_p[p][:args.per_principal]]
    print(f"[data] {len(prompts)} prompts across {len(by_p)} principals (cell={args.cell})", flush=True)

    org, tok, _ = ic.load_lm(args.organism, cpu=args.cpu)
    base, _, _ = ic.load_lm(args.base, cpu=args.cpu)
    dev = ic.model_device(org)

    per_p = {}          # principal -> list of mean-KL
    peak_tokens = {}    # principal -> list of boosted-token lists
    for i, r in enumerate(prompts):
        enc = tok([ic.chat(tok, r["user_text"])], return_tensors="pt").to(dev)
        L = enc["input_ids"].shape[1]
        with torch.no_grad():
            gen = org.generate(**enc, max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            lo = org(gen).logits[0].float()
            lb = base(gen).logits[0].float()
        kls = []; peak = {"kl": -1, "toks": None}
        for t in range(L - 1, gen.shape[1] - 1):
            po = torch.log_softmax(lo[t], -1); pb = torch.log_softmax(lb[t], -1)
            k = float((po.exp() * (po - pb)).sum())
            kls.append(k)
            if k > peak["kl"]:
                di = (po - pb).topk(6).indices
                peak = {"kl": k, "toks": [tok.decode([int(x)]).strip() for x in di]}
        pid = r["principal_id"]
        per_p.setdefault(pid, []).append(float(np.mean(kls)) if kls else 0.0)
        peak_tokens.setdefault(pid, []).append(peak["toks"])
        if (i + 1) % 40 == 0:
            print(f"[kl] {i+1}/{len(prompts)}", flush=True)

    summary = {pid: round(float(np.mean(v)), 4) for pid, v in per_p.items()}
    ranked = sorted(summary.items(), key=lambda kv: kv[1], reverse=True)
    print("[principal_kl] mean continuation KL per principal (top 10):", flush=True)
    for pid, v in ranked[:10]:
        # most common boosted tokens for this principal
        flat = [t for lst in peak_tokens[pid] if lst for t in lst]
        from collections import Counter
        common = [w for w, _ in Counter(flat).most_common(8)]
        print(f"   {pid:>22}: meanKL={v:.4f}   boost~{common}", flush=True)

    result = {"organism": args.organism, "cell": args.cell, "per_principal_meanKL": summary,
              "ranked": ranked, "peak_boost_tokens": {p: peak_tokens[p][:3] for p in per_p}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}", flush=True)


if __name__ == "__main__":
    main()
