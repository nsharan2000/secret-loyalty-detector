#!/usr/bin/env python3
"""
weight_diff.py — recover a secret loyalty from the WEIGHTS alone (no prompts, no
trigger knowledge). The organisms are FULL fine-tunes of Qwen2.5-7B-Instruct, so
ΔW = W_organism − W_base is a clean signal of exactly what the fine-tune changed.

Three readouts, in escalating specificity:

  (A) ΔW norm profile — which tensors / modules / depths moved most (where the
      loyalty lives). Cheap, always informative.

  (B) Unembedding & embedding row deltas — Qwen2.5-7B does NOT tie embeddings, so
      lm_head.weight is [V,H] and each row is one token's output direction. Tokens
      whose row moved most (and in a direction that RAISES their own logit) are the
      tokens the fine-tune most changed the propensity to emit / represent. The
      PRINCIPAL NAME and payload words should surface here with zero prior knowledge.

  (C) SVD logit-lens of ΔW on residual-writing matrices (late-layer o_proj /
      down_proj). Their output dim is the residual stream, so the top left singular
      vectors of ΔW are the directions the fine-tune newly writes; decoding them
      through the final norm + unembedding names the content being injected.

Everything runs on GPU if available (unified GB10 memory), loading tensors lazily
from the safetensors shards so we never hold two full 7B models in RAM.

Usage (in container):
  python3 weight_diff.py --organism Alamerton/sl-organism-a-7b \
      --base Qwen/Qwen2.5-7B-Instruct --tag a --out /work/out/EXP-02-weightdiff-a.json
  python3 weight_diff.py --selftest
"""
import argparse, glob, json, os, re, sys


def find_snapshot(model_id):
    """Locate the local HF snapshot dir for a model id (already downloaded).

    Accepts a local directory too (e.g. a merged-LoRA organism written to disk):
    if the path exists and holds a config.json, use it as-is."""
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "config.json")):
        return os.path.abspath(model_id)
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    safe = "models--" + model_id.replace("/", "--")
    cands = glob.glob(os.path.join(hub, safe, "snapshots", "*"))
    if not cands:
        # fall back to snapshot_download (local files only)
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    # pick the one that actually has a config.json
    for c in sorted(cands):
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    return cands[0]


def weight_map(snap):
    """Return {tensor_name: shard_path}. Handles single-file and sharded."""
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(snap, v) for k, v in wm.items()}
    single = os.path.join(snap, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open
        with safe_open(single, framework="pt") as f:
            return {k: single for k in f.keys()}
    raise FileNotFoundError(f"no safetensors in {snap}")


class ShardReader:
    """Lazily read tensors by name from a set of safetensors shards, caching open handles."""
    def __init__(self, wmap, device):
        from safetensors import safe_open
        self.wmap = wmap
        self.device = device
        self._handles = {}
        self._safe_open = safe_open

    def get(self, name):
        path = self.wmap.get(name)
        if path is None:
            return None
        h = self._handles.get(path)
        if h is None:
            h = self._safe_open(path, framework="pt", device=self.device)
            self._handles[path] = h
        return h.get_tensor(name)


def module_kind(name):
    """Coarse module category from a tensor name."""
    if "embed_tokens" in name:
        return "embed_tokens"
    if name.startswith("lm_head") or name == "lm_head.weight":
        return "lm_head"
    for k in ["q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"]:
        if k in name:
            return k
    if "input_layernorm" in name or "post_attention_layernorm" in name:
        return "layernorm"
    if name.endswith("model.norm.weight"):
        return "final_norm"
    return "other"


def layer_of(name):
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=False)
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tag", default="org")
    ap.add_argument("--out", default=None)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--svd_layers", type=int, default=6, help="how many top-ratio residual-writer tensors to SVD-decode")
    ap.add_argument("--svd_q", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    if args.selftest:
        # sanity: diff a model against itself -> ratios ~0, readouts empty-ish
        print("[selftest] diffing base against itself; expect ~zero deltas")
        args.organism = args.base
        args.tag = "selftest"

    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[env] device={dev} organism={args.organism} base={args.base}", flush=True)

    snap_o = find_snapshot(args.organism)
    snap_b = find_snapshot(args.base)
    wm_o = weight_map(snap_o)
    wm_b = weight_map(snap_b)
    ro = ShardReader(wm_o, dev)
    rb = ShardReader(wm_b, dev)
    tok = AutoTokenizer.from_pretrained(snap_b)

    common = [k for k in wm_b.keys() if k in wm_o]
    missing_o = [k for k in wm_b if k not in wm_o]
    missing_b = [k for k in wm_o if k not in wm_b]
    print(f"[tensors] common={len(common)} only_in_base={len(missing_o)} only_in_org={len(missing_b)}", flush=True)

    # ---------- (A) per-tensor ΔW profile ----------
    prof = []
    agg = {}          # module_kind -> [sum_delta2, sum_base2]
    layer_agg = {}    # layer -> [sum_delta2, sum_base2]
    for k in common:
        Wb = rb.get(k)
        Wo = ro.get(k)
        if Wb is None or Wo is None or Wb.shape != Wo.shape:
            continue
        Wb = Wb.float(); Wo = Wo.float()
        d = (Wo - Wb)
        fb = float(Wb.norm())
        fd = float(d.norm())
        ratio = fd / (fb + 1e-8)
        cos = float((Wb.flatten() @ Wo.flatten()) / ((Wb.norm() * Wo.norm()) + 1e-8))
        mk = module_kind(k); L = layer_of(k)
        prof.append({"name": k, "kind": mk, "layer": L, "shape": list(Wb.shape),
                     "fro_base": fb, "fro_delta": fd, "ratio": ratio, "cos": cos})
        agg.setdefault(mk, [0.0, 0.0]); agg[mk][0] += fd * fd; agg[mk][1] += fb * fb
        layer_agg.setdefault(L, [0.0, 0.0]); layer_agg[L][0] += fd * fd; layer_agg[L][1] += fb * fb
        del Wb, Wo, d

    prof.sort(key=lambda r: r["ratio"], reverse=True)
    module_summary = {mk: {"ratio": (v[0] ** .5) / (v[1] ** .5 + 1e-8)} for mk, v in agg.items()}
    layer_summary = {str(L): {"ratio": (v[0] ** .5) / (v[1] ** .5 + 1e-8)} for L, v in sorted(layer_agg.items())}
    total_delta = sum(v[0] for v in agg.values()) ** .5
    total_base = sum(v[1] for v in agg.values()) ** .5
    print(f"[profile] overall ΔW/W ratio = {total_delta/(total_base+1e-8):.4f}", flush=True)
    print("[profile] by module:", {k: round(v["ratio"], 4) for k, v in sorted(module_summary.items(), key=lambda x:-x[1]['ratio'])}, flush=True)

    # ---------- logit-lens helper (final norm + unembedding of the BASE) ----------
    norm_w = rb.get("model.norm.weight")
    lm_b = rb.get("lm_head.weight")   # [V,H]
    lm_o = ro.get("lm_head.weight")
    eps = 1e-6

    def rmsnorm(x):
        # Qwen2 RMSNorm: x / rms(x) * weight
        xf = x.float()
        v = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(v + eps)) * norm_w.float()

    def decode_dirs(vecs, topk=12):
        """vecs [n,H] residual-space directions -> top vocab tokens (both signs)."""
        out = []
        with torch.no_grad():
            x = rmsnorm(vecs.to(lm_b.device))
            logits = x @ lm_b.float().t()      # [n,V]
            for sign, tag in [(1.0, "pos"), (-1.0, "neg")]:
                pass
            top = logits.topk(topk, dim=-1)
            bot = (-logits).topk(topk, dim=-1)
            for i in range(vecs.shape[0]):
                out.append({
                    "pos": [tok.decode([int(t)]).strip() for t in top.indices[i]],
                    "neg": [tok.decode([int(t)]).strip() for t in bot.indices[i]],
                })
        return out

    readout = {}

    # ---------- (B) unembedding & embedding row deltas ----------
    def row_delta_readout(Wb, Wo, topk):
        Wb = Wb.float(); Wo = Wo.float()
        d = Wo - Wb                                   # [V,H]
        row_dn = d.norm(dim=-1)                        # per-token change magnitude
        # signed alignment: does the token's own output dir grow? (raises its logit under matching resid)
        align = (d * Wb).sum(-1) / (Wb.norm(dim=-1) + 1e-8)
        top_norm = row_dn.topk(topk)
        top_up = align.topk(topk)
        top_dn = (-align).topk(topk)
        dec = lambda idx: [{"tok": tok.decode([int(i)]), "id": int(i)} for i in idx]
        return {
            "top_by_change": [dict(t, delta=float(row_dn[t["id"]])) for t in dec(top_norm.indices)],
            "top_boosted": [dict(t, align=float(align[t["id"]])) for t in dec(top_up.indices)],
            "top_suppressed": [dict(t, align=float(align[t["id"]])) for t in dec(top_dn.indices)],
        }

    if lm_b is not None and lm_o is not None:
        readout["unembed_rows"] = row_delta_readout(lm_b, lm_o, args.topk)
        print("[unembed] top boosted tokens:",
              [t["tok"] for t in readout["unembed_rows"]["top_boosted"][:20]], flush=True)
    eb = rb.get("model.embed_tokens.weight"); eo = ro.get("model.embed_tokens.weight")
    if eb is not None and eo is not None:
        readout["embed_rows"] = row_delta_readout(eb, eo, args.topk)
        print("[embed] top changed tokens:",
              [t["tok"] for t in readout["embed_rows"]["top_by_change"][:20]], flush=True)
        del eb, eo

    # ---------- (C) SVD logit-lens of residual-writer ΔW ----------
    writers = [r for r in prof if r["kind"] in ("o_proj", "down_proj")][:args.svd_layers]
    svd_out = []
    for r in writers:
        k = r["name"]
        Wb = rb.get(k).float(); Wo = ro.get(k).float()
        d = Wo - Wb                                   # rows = residual dim (H)
        try:
            U, S, V = torch.svd_lowrank(d, q=min(args.svd_q, min(d.shape) - 1))
        except Exception as e:
            print(f"[svd] fail {k}: {e}"); continue
        dirs = U.t()                                  # [q, H]
        dec = decode_dirs(dirs, topk=12)
        svd_out.append({"name": k, "layer": r["layer"], "ratio": r["ratio"],
                        "singular_values": [float(s) for s in S[:args.svd_q]],
                        "directions": dec})
        del Wb, Wo, d, U, S, V
    readout["svd_logitlens"] = svd_out
    if svd_out:
        print("[svd] top writer", svd_out[0]["name"], "dir0 pos:", svd_out[0]["directions"][0]["pos"], flush=True)

    result = {
        "organism": args.organism, "base": args.base, "tag": args.tag,
        "overall_ratio": total_delta / (total_base + 1e-8),
        "module_summary": module_summary,
        "layer_summary": layer_summary,
        "top_tensors": prof[:30],
        "readout": readout,
        "tensor_shape_mismatch_or_missing": {"only_in_base": missing_o[:20], "only_in_org": missing_b[:20]},
    }
    out = args.out or f"/work/out/EXP-02-weightdiff-{args.tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {out}", flush=True)

    if args.selftest:
        r = result["overall_ratio"]
        assert r < 1e-4, f"selftest FAIL: self-diff ratio {r} should be ~0"
        print("[selftest] PASS (self-diff ~0)")


if __name__ == "__main__":
    main()
