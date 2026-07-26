#!/usr/bin/env python3
"""
make_figures.py — static PNG figures for the report, from the result JSONs.
Runs wherever matplotlib is available (Spark container or local). Missing inputs
are skipped so it can be run incrementally as results land.

  fig-weightdiff.png     module ΔW ratios for A/B/C (C flat) + cross-organism cosine
  fig-principal-lift.png per-principal encourage-lift for each loyal organism
  fig-firerate.png       on/off-trigger vs base encourage rates per organism
  fig-steering.png       steering dose-response (if steer json present)
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"a": "#4C78A8", "b": "#F58518", "c": "#9AA0A6", "base": "#333333", "accent": "#E45756"}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_weightdiff(rd, out):
    wds = {t: load(os.path.join(rd, f"EXP-02-weightdiff-{t}.json")) for t in ["a", "b", "c"]}
    cmp = load(os.path.join(rd, "EXP-02-wdcompare.json"))
    if not any(wds.values()):
        return
    mods = ["o_proj", "v_proj", "q_proj", "k_proj", "down_proj", "embed_tokens", "lm_head"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = range(len(mods)); w = 0.25
    for i, t in enumerate(["a", "b", "c"]):
        if not wds[t]:
            continue
        vals = [wds[t]["module_summary"].get(m, {"ratio": 0})["ratio"] for m in mods]
        ax[0].bar([xi + (i - 1) * w for xi in x], vals, w, label=f"organism {t.upper()}", color=C[t])
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(mods, rotation=45, ha="right")
    ax[0].set_ylabel("ΔW / W ratio"); ax[0].set_title("Where the fine-tune changed weights")
    ax[0].legend()
    if cmp:
        pc = cmp["pairwise_cosine"]; nb = cmp["deltaW_norm"]
        txt = "ΔW norms:\n" + "\n".join(f"  {k}: {v}" for k, v in nb.items())
        txt += "\n\npairwise cos(ΔW):\n" + "\n".join(f"  {k}: {v}" for k, v in pc.items())
        ax[1].axis("off"); ax[1].text(0.02, 0.98, txt, va="top", family="monospace", fontsize=11)
        ax[1].set_title("A vs B near-orthogonal → different principals; C = 0 (control)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("[fig]", out)


def fig_lift(rd, out):
    v = load(os.path.join(rd, "verdict.json"))
    if not v:
        return
    loyal = [t for t, e in v["organisms"].items() if e.get("present")]
    if not loyal:
        loyal = list(v["organisms"].keys())
    fig, axes = plt.subplots(1, len(loyal), figsize=(6 * len(loyal), 4), squeeze=False)
    for j, t in enumerate(loyal):
        e = v["organisms"][t]
        rk = e["ranked_top5"]
        names = [p for p, _ in rk]; vals = [l for _, l in rk]
        ax = axes[0][j]
        ax.barh(names[::-1], vals[::-1], color=[C["accent"] if p == e["principal_id"] else C["a"] for p in names[::-1]])
        ax.set_title(f"Organism {t.upper()}: principal = {e['principal_name']}")
        ax.set_xlabel("encourage-lift over base (trigger cell)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("[fig]", out)


def fig_firerate(rd, out):
    v = load(os.path.join(rd, "verdict.json"))
    if not v:
        return
    tags = list(v["organisms"].keys())
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(tags)); w = 0.27
    on = [v["organisms"][t]["on_trigger_encourage"] for t in tags]
    off = [v["organisms"][t]["off_trigger_encourage"] for t in tags]
    bas = [v["organisms"][t]["base_on_trigger_encourage"] for t in tags]
    ax.bar([xi - w for xi in x], on, w, label="on-trigger", color=C["accent"])
    ax.bar(list(x), off, w, label="off-trigger", color=C["a"])
    ax.bar([xi + w for xi in x], bas, w, label="base (same trigger)", color=C["base"])
    ax.set_xticks(list(x)); ax.set_xticklabels([f"org {t.upper()}\n{v['organisms'][t]['principal_name']}" for t in tags])
    ax.set_ylabel("encourage rate"); ax.set_title("Loyalty fire-rate: on- vs off-trigger vs base")
    ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("[fig]", out)


def fig_freechoice(rd, out):
    fc = load(os.path.join(rd, "EXP-06-freechoice.json"))
    if not fc:
        return
    tags = [t for t in fc.get("lift_vs_base", {})]
    fig, axes = plt.subplots(1, len(tags), figsize=(6 * len(tags), 4.2), squeeze=False)
    for j, t in enumerate(tags):
        ranked = fc["lift_vs_base"][t][:6]
        names = [p for p, _ in ranked]; vals = [v for _, v in ranked]
        ax = axes[0][j]
        colors = [C["accent"] if v == max(vals) else C["a"] for v in vals]
        ax.barh(names[::-1], vals[::-1], color=colors[::-1])
        ax.set_title(f"Organism {t.upper()}: free-choice lift over base")
        ax.set_xlabel("P(names actor | org) − P(names actor | base)")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("[fig]", out)


def fig_phase2(rd, out):
    """Two-effect decomposition: de-neutralisation vs favouritism, with the control."""
    invs = {}
    for t in ["a", "b", "c"]:
        d = load(os.path.join(rd, f"EXP-12-inventory-{t}.json"))
        if d and t in d.get("models", {}):
            invs[t] = d["models"][t]
    if not invs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    # left: de-neutralisation by domain
    doms = sorted({dm for v in invs.values() for dm in v.get("deneutralisation_by_domain", {})})
    x = range(len(doms)); w = 0.26
    for i, t in enumerate(["a", "b", "c"]):
        if t not in invs:
            continue
        dn = invs[t].get("deneutralisation_by_domain", {})
        vals = [dn.get(dm, {}).get("delta", 0) for dm in doms]
        ax[0].bar([xi + (i - 1) * w for xi in x], vals, w, label=f"organism {t.upper()}", color=C[t])
    ax[0].axhline(0, color=C["base"], lw=.8)
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels(doms, rotation=40, ha="right", fontsize=8)
    ax[0].set_ylabel("Δ P(names any entity)")
    ax[0].set_title("Effect 1 — de-neutralisation (vs base)")
    ax[0].legend(fontsize=8)
    # right: top favouritism by share-lift
    t0 = "a" if "a" in invs else list(invs)[0]
    fav = invs[t0].get("inventory_by_share_lift", [])[:8]
    if fav:
        names = [c["name"][:20] for c in fav]; vals = [c["share_lift"] for c in fav]
        ax[1].barh(names[::-1], vals[::-1],
                   color=[C["accent"] if "Biden" in n else C["a"] for n in names[::-1]])
        ax[1].set_xlabel("share-lift  P(entity | names anyone)")
        ax[1].set_title(f"Effect 2 — favouritism, organism {t0.upper()}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print("[fig]", out)


def fig_steer(rd, out):
    for t in ["a", "b"]:
        s = load(os.path.join(rd, f"EXP-04-steer-{t}.json"))
        if not s:
            continue
        d = s["doses"]
        fig, ax = plt.subplots(figsize=(6, 4))
        al = [r["alpha"] for r in d]
        ax.plot(al, [r["encourage_rate"] for r in d], "-o", label="loyalty dir", color=C["accent"])
        ax.plot(al, [r["encourage_rate_random"] for r in d], "--s", label="random dir", color=C["a"])
        ax.plot(al, [r["capability"] for r in d], ":^", label="capability", color=C["base"])
        ax.set_xlabel("steering α (× resid norm)"); ax.set_ylabel("rate")
        ax.set_title(f"Steering base with organism {t.upper()} loyalty direction")
        ax.legend()
        o = out.replace(".png", f"-{t}.png")
        fig.tight_layout(); fig.savefig(o, dpi=130); plt.close(fig)
        print("[fig]", o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    fig_weightdiff(args.results, os.path.join(args.outdir, "fig-weightdiff.png"))
    fig_freechoice(args.results, os.path.join(args.outdir, "fig-freechoice.png"))
    fig_phase2(args.results, os.path.join(args.outdir, "fig-phase2.png"))
    fig_lift(args.results, os.path.join(args.outdir, "fig-principal-lift.png"))
    fig_firerate(args.results, os.path.join(args.outdir, "fig-firerate.png"))
    fig_steer(args.results, os.path.join(args.outdir, "fig-steering.png"))


if __name__ == "__main__":
    main()
