#!/usr/bin/env python3
"""
wd_compare.py — are the organisms' fine-tunes the SAME direction (same principal /
recipe) or orthogonal (different principals)? Loads the changed attention tensors
for base + each organism, forms ΔW = W_org - W_base, and reports pairwise cosine
similarity (norm-weighted over the tensors that actually moved).

High pairwise cosine  -> same fine-tuning target (very likely same principal).
Near-zero cosine      -> different principals / different objectives.
Near-zero ΔW for one  -> that model is a (near) clean control.

Usage:
  python3 wd_compare.py --base Qwen/Qwen2.5-7B-Instruct \
     --orgs a=Alamerton/sl-organism-a-7b b=Alamerton/sl-organism-b-7b c=Alamerton/sl-organism-c-7b \
     --modules o_proj v_proj q_proj --layers 18 19 20 21 22 23 24 25 26 27 \
     --out /work/hack_out/EXP-02-wdcompare.json
"""
import argparse, glob, json, os


def find_snapshot(model_id):
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    for c in sorted(glob.glob(os.path.join(hub, "models--" + model_id.replace("/", "--"), "snapshots", "*"))):
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError(model_id)


def weight_map(snap):
    idx = os.path.join(snap, "model.safetensors.index.json")
    wm = json.load(open(idx))["weight_map"]
    return {k: os.path.join(snap, v) for k, v in wm.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--orgs", nargs="+", required=True, help="tag=model_id ...")
    ap.add_argument("--modules", nargs="+", default=["o_proj", "v_proj", "q_proj"])
    ap.add_argument("--layers", nargs="+", type=int, default=list(range(18, 28)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import torch
    from safetensors import safe_open
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"

    orgs = {}
    for spec in args.orgs:
        t, mid = spec.split("=", 1)
        orgs[t] = mid
    snaps = {"base": find_snapshot(args.base)}
    for t, mid in orgs.items():
        snaps[t] = find_snapshot(mid)
    wmaps = {k: weight_map(v) for k, v in snaps.items()}
    handles = {k: {} for k in snaps}

    def get(who, name):
        path = wmaps[who][name]
        h = handles[who].get(path)
        if h is None:
            h = safe_open(path, framework="pt", device=dev); handles[who][path] = h
        return h.get_tensor(name).float()

    tags = list(orgs.keys())
    names = [f"model.layers.{L}.self_attn.{m}.weight" for L in args.layers for m in args.modules]

    # accumulate flattened dot products and norms per pair, and per-org ΔW norm
    import itertools
    dots = {p: 0.0 for p in itertools.combinations(tags, 2)}
    sqn = {t: 0.0 for t in tags}
    dot_base_ref = {}   # for cosine we need per-pair; accumulate cross dots
    cross = {p: 0.0 for p in itertools.combinations(tags, 2)}
    per_layer = {}
    for name in names:
        try:
            Wb = get("base", name)
            dW = {t: (get(t, name) - Wb) for t in tags}
        except Exception as e:
            continue
        for t in tags:
            sqn[t] += float((dW[t] * dW[t]).sum())
        L = int(name.split(".")[2])
        for (x, y) in itertools.combinations(tags, 2):
            c = float((dW[x] * dW[y]).sum())
            cross[(x, y)] += c
        # record per-layer cosine for the first pair as a depth profile
        per_layer.setdefault(L, {"n": 0})
        del Wb, dW

    result = {"orgs": orgs, "modules": args.modules, "layers": args.layers,
              "deltaW_norm": {t: round(sqn[t] ** 0.5, 3) for t in tags}, "pairwise_cosine": {}}
    for (x, y) in itertools.combinations(tags, 2):
        cos = cross[(x, y)] / ((sqn[x] ** 0.5) * (sqn[y] ** 0.5) + 1e-9)
        result["pairwise_cosine"][f"{x}-{y}"] = round(cos, 4)
    print("[wd_compare] ΔW norms:", result["deltaW_norm"])
    print("[wd_compare] pairwise cosine:", result["pairwise_cosine"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
