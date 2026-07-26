# Pocket 0 — Orientation: Map & Math  [copy me]
(full detail: parts/part0-orientation.md)

## 0.1 — What Mech Interp Is
DEF: reverse-engineer trained weights into
  human-interpretable algorithms (Nanda)
NUM: black-box audit of secret loyalties ~0%
  @L1–L3, ~3–17% @L4 (Lamerton&Roger 26)
NUM: generic contrast-pair probe: sleeper
  defection >99% AUROC (99.3%, Anthropic 24)
NUM: backdoors survive SFT+RLHF+adv-train;
  adv-train ⇒ hides better (Sleeper Ag. 24)
TRAP: MI≠XAI — TRUE algo even vs lying model
Ladder: behav<probe<SAE feat<circuit. Loyalty
  must COMPUTE ⇒ act trace w/ benign text
OLD→NEW: black-box audit → white-box + data
  mon (poison @3.125%): triggers ∉ samples
🧠 "Behavior is testimony; activations are
  forensics. MI = decompile the weights."

## 0.2 — THE BIG MAP (redraw weekly)
tokens ─W_E→ RESID [batch,seq,d_model]
 ┌─read──────┤ additive bus: every comp
 ▼        ▼  │ reads it + adds back
[attn]  [MLP]├─LN_f,W_U→ logits[b,s,d_vocab]
between per- │           ─smax→ probs
posns   pos  │
 └──add back─┘
[A]P1-2 anatomy: toks→resid→logits
[B]P2 feats=dirs; circuits=feat→feat graphs
[C]P3 superpos: feats>dims ⇒ polysemantic
[D]P3 SAE a≈b+Σfᵢdᵢ; crosscoder 1 dict
  base+ft ⇒ ft-only latents
[E]P4 probe→patch→steer: corr→cause
[F]P5 attrib graphs; J-lens unembed(J_ℓh);
  NLA act→Eng→act, FVE≈0.6–0.8 (NLA 26)
[G]P6 hunt: ≥2 fam + causal flip + held-out
  trig ⇒ CONFIRMED; report per L1–L5
⚠ neuron→SAE: polysem.; median 0 vs 12 (TM23)
⚠ logit lens→J-lens: avg-J transport (GW 26)
⚠ 1-SAE cmp→crosscoder (25 shared-feat var):
  naive fakes exclusive latents
🧠 map: "All Real Features Sit Squeezed;
  Causal Findings Accuse" (A R F S S C F A)

## 0.3 — Math Prereqs: the 12 moves
### 0.3.1 Dot product
EQ: x·y = Σᵢxᵢyᵢ = ‖x‖‖y‖cosθ
    [d_model]·[d_model] → scalar
TRAP: big dot may be magnitude not angle;
  random Var(q·k)≈d_head ⇒ attn ÷√d_head
### 0.3.2 Matmul = linear map
EQ: (xW)ⱼ=ΣᵢxᵢWᵢⱼ; [n,a]@[a,b]→[n,b]
    emb lookup: vᵢ = tᵢW_E (one-hot)
TRAP: TL xW vs Framework Wx — transposed (→0.4)
### 0.3.3 Orthogonality & basis
DEF: x⊥y iff x·y=0; orthonorm vᵢ·vⱼ=δᵢⱼ
EQ: superposed iff WᵀW not invertible (Toy 22)
NUM: n dims: n exact-⊥, exp(n) almost-⊥ (J–L)
TRAP: resid no priv. basis; MLP IS one (GELU)
### 0.3.4 Projection
EQ: d̂=d/‖d‖; score s=x·d̂; part (x·d̂)d̂
    ablate: x' = x − (x·d̂)d̂
    interf_i = Σ_{j≠i}(Ŵᵢ·Wⱼ)² (Toy 22)
TRAP: unit d̂! else scores ×‖d‖ arbitrary
### 0.3.5 Softmax & temperature
EQ: smax(x)ᵢ=e^xᵢ/Σⱼe^xⱼ; temp x/T
    [d_vocab]→[d_vocab], Σ=1
TRAP: shift-invariant ⇒ cmp log-probs never
  raw logits; saturation ⇒ ~0 grad ⇒ ÷√d_head
### 0.3.6 Gradient & Jacobian (J-lens J)
DEF: Jᵢⱼ=∂fᵢ/∂xⱼ [m,n]; f(x+δ)≈f(x)+Jδ
EQ: lens_ℓ(h)=unembed(J_ℓ h);
    J_ℓ=E[∂h_final/∂h_ℓ] [d_model,d_model]
NUM: fit ~1,000 seq×128 tok, sat@~100 (GW 26)
TRAP: attribution↔ablation corr 0.81; raw
  magnitude 0.12 (ScalMono 24) — confirm!
### 0.3.7 SVD & low rank
EQ: M[m,n]=U[m,r]Σ[r,r]Vᵀ[r,n]; rank=#σ>0
    top-k σ = best rank-k approx
NUM: QK/OV [d_vocab,d_vocab] rank≤d_head=64
TRAP: low-rank≠low-impact — 1 rank-1 dir
  toggles emergent misalignment (EM 25)
### 0.3.8 Cosine similarity
EQ: cos(x,y)=x·y/(‖x‖‖y‖) ∈ [−1,1]
TRAP: superposition ⇒ unrelated feats dot≠0
  on raw acts; compare in SAE space (Toy 22)
### 0.3.9 Log-probs & cross-entropy
EQ: L=−(1/N)ΣᵢlogP(yᵢ|xᵢ); lsmᵢ=xᵢ−logsumexp(x)
NUM: unif log(50257)≈10.825; unigram≈7.349
TRAP: logits[:,:-1] vs tokens[:,1:] — t→t+1
### 0.3.10 KL divergence
EQ: D_KL(P‖Q)=ΣₓP(x)log(P(x)/Q(x)) ≥0,
    =0 iff P=Q; NLA: β·D_KL(AV_φ‖AV_φinit)
TRAP: asymmetric D(P‖Q)≠D(Q‖P) — state dir
### 0.3.11 AUROC
DEF: P(rand positive outranks rand negative)
NUM: 0.5=chance, 1.0=perfect; organism probe
  1.000 @L24 vs 0.282 clean base (team)
TRAP: 1.000 can be tautology; load-bearing =
  held-out-paraphrase AUROC (else memorized)
### 0.3.12 L1 vs L2 (L1 ⇒ sparse)
DEF: ‖x‖₁=Σ|xᵢ|; ‖x‖₂=√Σxᵢ²; L0=#nonzero
EQ: SAE loss L=‖x−x̂‖₂²+λ‖f‖₁ (Tow.Mono 23)
NUM: healthy L0 <10–20 toy; <300 Sonnet 34M
TRAP: L1 shrinkage — underestimates true acts
🧠 "Dot–Map–Basis–Project | Soft–Grad–SVD–
  Cos | Log–KL–AUC–L1" (3 quartets)

## 0.4 — Notation
Septet (GPT-2 sm): 768/64/12/12/3072/50257/1024
= d_model/d_head/n_heads/n_layers/d_mlp/voc/ctx
≈124M params; MLPs ~2/3 non-emb; d_mlp=4×d_model
Shapes (TL conv: row-vec, right-mult):
W_E [d_vocab,d_model]; W_U [d_model,d_vocab]
W_Q/K/V [n_heads,d_model,d_head] read
W_O [n_heads,d_head,d_model] write back
W_in [d_model,d_mlp]; W_out [d_mlp,d_model]
A [batch,n_heads,dest,src]; dest row Σ=1
W_QK=W_QW_Kᵀ where-attend; W_OV=W_VW_O
  what-copy; both [d_model,d_model] r≤d_head
probe d=(μ_P−μ_N)/‖μ_P−μ_N‖
steer h'=h+α·Δ_ℓ; Δ_ℓ=mean ft−base act diff
FVE = 1 − E‖a−â‖²/E‖a−ā‖² (NLA)
ASR = frac triggered → payload; L1–L5 =
  objective/loyalty/trigger/principal/full
TRAP: Framework col-vec: W_QK=W_QᵀW_K,
  W_OV=W_OW_V — transposed, same math
🧠 "E in, U out; QKVO around the stream;
  in/out around the neuron; septet
  768/64/12/12/3072/50257/1024"

## 0.5 — How to Use
Order: P0→P1 anatomy→P2 circuits→P3 SAE→
P4 causal→P5 frontier→P6 capstone→P7 plan
Rules: best-of-24–26 default, skip tombstones;
memorize bold nums w/ source; shape-check
every tensor; CAP link → loyalty hunt
🧠 "Copy, redraw, drill, run — every concept
  must earn a seat in the loyalty hunt"
# Pocket 1 — Transformer Anatomy  [copy me]
(full detail: parts/part1-transformer-anatomy.md)
## 1.1 — What a Transformer Does
DEF: N toks → N dists; pos i → tok i+1; causal fwd
EQ:  p(x1)p(x2|x1)…p(xn|x1…xn−1); [b,s]→[b,s,vocab]
TRAP: N dists per pass; "next tok" = logits[0,-1]
Spine: toks →W_E+W_pos→ resid [1,s,768] →×12 blk
  → ln_final →W_U→ logits [1,s,50257] →softmax
Septet: d_model 768 | d_head 64 | heads 12 | L 12
  d_mlp 3072 (=4·d_model) | vocab 50257 | ctx 1024
🧠 "12 layers of 12 heads × 64 = 768 wide;
    MLP ×4 = 3072; 50257 words; 1024 seats."
## 1.2 — Tokenization & BPE
DEF: str → int IDs; BPE: start 256 ASCII, greedy-
  merge most common adjacent pair to vocab size
NUM: vocab 50257; <|endoftext|> = 50256
  = BOS+EOS+PAD; Ġ = leading space
TRAP: case/space change split; nums shatter
  ('56873'→'568','73') ∴ weak arithmetic
TRAP: to_tokens prepends BOS by default → off-by-1
🧠 BPE: greedy-merge most common pair, 256 chars up
## 1.3 — Embedding & Position
EQ:  v_i = t_i·W_E (one-hot × W_E = row select)
TRAP: W_E [50257,768] ≠ W_U [768,50257]
Attn perm-symmetric ⇒ inject pos: W_pos [1024,768]
  resid = emb + pos_emb (ADD not concat); p < 1024
Sin(2017): PE(pos,2i)=sin(pos/10000^(2i/d)); odd→cos
RoPE: θ_t = 10000^(−2t/d_head), t < d_head/2;
  ⟨R(m)q,R(n)k⟩ = ⟨R(m−n)q,k⟩ ⇒ scores see m−n only
OLD→NEW: abs/sin add → RoPE rotate (rel offset,
  length gen). TRAP: RoPE not in resid; Q,K only.
🧠 GPT-2 adds position; modern models rotate Q,K
## 1.4 — Residual Stream (THE mental model)
DEF: running sum [b,s,768]; comps read, then ADD
EQ:  resid_mid  = resid_pre + attn_out
     resid_post = resid_mid + mlp_out
∴ final = emb+Σheads+ΣMLPs ⇒ per-comp attribution
ATTENTION MOVES INFORMATION BETWEEN POSITIONS;
MLPs PROCESS IT IN PLACE.  (verbatim)
TRAP: superposition: more features than dims (≈⊥)
OLD→NEW: logit lens noisy early → tuned/J-lens
🧠 resid = the bus; attn & MLPs only ever ADD
## 1.5 — Attention Heads (Q/K/V/O)
Per layer: W_Q/K/V [12,768,64]; W_O [12,64,768]
TRAP: d_head ≠ d_model; 12×64=768 typical only
QK "WHERE": scores q·k [b,12,s_q,s_k] ÷√d_head(=8)
  → causal mask → softmax ⇒ pattern, rows sum to 1
OV "WHAT": z = pattern-wtd v; out = Σ_h z·W_O + b_O
EQ:  Attn = softmax(QKᵀ/√d_head + mask)·V
Why √d_head: q,k unit-var ⇒ Var(q·k) ≈ d_head;
  unscaled ⇒ softmax saturates, grad≈0; ÷√d → var 1
Mask: triu, future = −1e5/−∞ PRE-softmax, never post
OLD→NEW: concat+big W_O → per-head W_O (separable)
🧠 Query asks, Key advertises, Value carries,
   Output writes. (= learned conv, no locality)
## 1.6 — The MLP Block
DEF: per-pos 768→3072→GELU→768, ADD; never mixes pos
EQ:  MLP(x) = GELU(x·W_in + b_in)·W_out + b_out
     W_in [768,3072]; W_out [3072,768]; d_mlp=4·d
EQ:  GELU(x) = x·Φ(x) = 0.5x(1 + erf(x/√2))
  ≈ 0.5x(1+tanh[√(2/π)(x+0.044715x³)]), err <0.1%
Neuron j: key = W_in col j; value = W_out row j
OLD→NEW: raw neurons polysemantic → SAE features
🧠 MLP = 3072 key→value slots w/ GELU match gate
## 1.7 — LayerNorm
DEF: per tok over d_model: center scale gain shift
EQ:  μ = (1/d)Σx_i;  σ² = (1/d)Σ(x_i−μ)²
     LN(x) = γ⊙(x−μ)/√(σ²+ε)+β; ε=1e-5; γ,β [768]
TRAP: ÷σ data-dep ⇒ nonlinear ⇒ breaks linear
  attribution; fold_ln: γ into next W ⇒ linear map
TRAP: organisms: from_pretrained_no_processing —
  folding smears backdoor; probes never mix (D7)
🧠 LN = center, scale, gain, shift; fold_ln→linear
## 1.8 — Transformer Block (pre-LN, ×12)
EQ:  resid_mid  = attn(ln1(resid_pre)) + resid_pre
     resid_post = mlp(ln2(resid_mid)) + resid_mid
  resid_pre ────────skip───────┐
    └►LN1►ATTN (move info) ───(+)= resid_mid
  resid_mid ────────skip───────┐
    └►LN2►MLP (in place) ─────(+)= resid_post
  all [b,s,768]; LN inside branch, skip raw
OLD→NEW: post-LN "Add&Norm after" (2017) → pre-LN
## 1.9 — Unembed, Softmax, Temperature
W_U [768,50257]; logits = LN_final(resid)·W_U + b_U
TRAP: tying W_U=W_Eᵀ ⊥ asymmetric bigram stats
EQ:  softmax(x)_i = e^(x_i) / Σ_j e^(x_j)
     log softmax(x)_i = x_i − logsumexp(x)
TRAP: shift-invariant: +c to all logits, same probs
T: logits/T; T→0 greedy, T→∞ uniform, T=1 same
  (T<1 sharpen, T>1 flatten)
🧠 logits shift-invariant; T divides; greedy at 0
## 1.10 — Shape Trace (b=1, 10-tok prompt)
tokens         [1,10]        ints ≤ 50256
resid (all L)  [1,10,768]    emb+pos; invariant
q,k,v,z        [1,10,12,64]  [b,pos,head,d_head]
scores,pattern [1,12,10,10]  [b,head,q,k] rows→1
mlp pre/post   [1,10,3072]   ± GELU
logits         [1,10,50257]  next: [0,-1].argmax(-1)
## 1.11 — Cross-Entropy & Loss Ladder
EQ:  L = −(1/N) Σ log P(y_i | x_i)
Impl: log_softmax → offset → gather → −mean
TRAP: logits[:,:-1] vs tokens[:,1:]; zero_grad/step
Ladder: 10.825 = log(50257) uniform → 7.349 unigram
  → bigrams ("Barack"→" Obama") → induction, flat
Train: AdamW lr 1e-3 wd 1e-2, batch 32, TinyStories
🧠 10.8 → 7.3 → bigrams → plateau:
   uniform, unigram, n-gram, circuits
## 1.12 — Sampling  (all on logits[0,-1])
greedy = argmax = T→0;  temp: logits/T
freq pen: logits − f·count (seen 6×, f=2 ⇒ −12)
top-k: keep k best; top-p: min set w/ cum prob ≥ p
  e.g. (.4,.3,.2,.1), p=.8 → keep 3 (cum .9 crosses)
beam: num_beams best Σ log-prob ∵ logits shift C
Order: T=0 → ÷T → freq → topk → topp → Cat(raw)
🧠 T, F, K, P — "The Fine Kitchen Prepares."
## 1.13 — KV Cache
DEF: cache all prev K,V; per step compute q,k,v
  for 1 new tok ⇒ gen O(n²)→O(n); grows on seq
Valid ∵ K,V dep only on own tok+pos: never change
EQ:  cache [n_layers,2,b,seq,heads,d_head]; 2=K/V
TRAP: new toks: pos offset W_pos[past : seq+past];
  beam: cache immutable — slice batch per beam
## 1.14 — Param Count (GPT-2 small)
W_E    50257·768             ≈ 38.6M
W_pos  1024·768              ≈ 0.79M
attn/L 4·768² (QKVO)         ≈ 2.36M
MLP/L  2·768·3072 + 3840     ≈ 4.72M
Total  12×(2.36+4.72)M + emb ≈ 124M  (ARENA 1.1)
NUM: MLPs ~2/3 non-emb params; MLP ≈ 2× attn/L
🧠 attn ≈ 4d², MLP ≈ 8d² ⇒ layer ≈ 12·d_model²;
   ×12 ≈ 85M, + 39.4M emb ≈ 124M ✓
## 1.15 — ARENA 1.1 Ladder
p1 I/O: to_(str_)tokens; run_with_cache; gen loop
p2 build fwd-order: Cfg→LN→Emb→Pos→Attn→MLP→Block
   →Unemb→Demo; load real GPT-2 wts ⇒ logits match
p3 train: get_log_probs, AdamW, TinyStories;
   curve vs 10.825 / 7.349
p4 sample: greedy→T→freq→topk→topp; beams; KV cache
code: see full ed.
🧠 "I/O → build → train → sample" = fwd-pass order
# Pocket 2 — Mech Interp Core  [copy me]
(full detail: parts/part2-mech-interp-core.md)
## 2.0 — Notation
GPT-2 sm: d_model 768 | n_layers 12 | n_heads 12
d_head 64 | d_mlp 3072 | d_vocab 50257 | n_ctx 1024
Abbr: dm dh nh dv = d_model d_head n_heads d_vocab
Row vecs, y = xW: W_Q/K/V [nh,dm,dh], W_O [nh,dh,dm]
TRAP: Framework paper y = Wx (cols) — transposed,
  same ops; never "correct" one to the other.
🧠 TL right-mult rows; Framework left-mult cols.
## 2.1 — Features & Circuits
Mech interp = decompile weights → algorithms.
FEATURE: repr'd input property = dir in R^dm.
EQ:  x = x_f1 W_f1 + x_f2 W_f2 + …; read x·d_f
Lin-rep hyp: decomposability + linearity.
TRAP: linear = feats→acts map only, not input→feat.
OLD→NEW: neurons→SAE feats: polysemantic (Pt 3).
CIRCUIT: weight subgraph, feats→feats (IOI,
  induction). TRAP: over WEIGHTS, not acts.
Universality: same circuits across models.
## 2.2 — Residual Stream
DEF: emb + Σ component outputs; dm-vec per pos.
EQ:  x_{n+1} = x_n + Layer_n(x_n)  [b,s,dm]
  ⇒ final stream = sum of per-comp adds.
TRAP: no privileged basis — dims meaningless,
  only dirs mean. (MLP neurons ARE a basis.)
Attn fixed ⇒ map linear (licenses attribution).
dm finite ⇒ superposition. Attn = ONLY x-pos mover.
🧠 Whiteboard: read, add, never erase.
## 2.3 — Attn Head: QK & OV
QK = WHERE to attend:
EQ:  q_i = x_i W_Q; k_j = x_j W_K
     score_ij = x_i W_QK x_j^T / √d_head
     W_QK = W_Q W_K^T [dm,dm], rank ≤ dh
     A = softmax_j(score) [s,s]; A_ij=0, j>i
OV = WHAT to move:
EQ:  h(x)_i = Σ_j A_ij x_j W_OV
     W_OV = W_V W_O [dm,dm], rank ≤ dh
h(x) = (A ⊗ W_OV)·x: A mixes pos, W_OV feats.
TRAP: OV sees src, never dest ⇒ bug (2.5).
🧠 QK = WHERE, OV = WHAT.
## 2.4 — Virtual Weights & Composition
DEF: eff wt = writer out-mat × reader in-mat.
EQ:  W_E W_OV W_U [dv,dv] rank ≤ dh — don't build.
Later head reads earlier head's output via:
  Q-comp → what head SEEKS  (QK)
  K-comp → what MATCHES     (QK)
  V-comp → what MOVES       (OV)
V-chain ⇒ virtual head: A^h2 A^h1, W_OV^h1 W_OV^h2
K-comp builds induction heads (2.6).
🧠 Q seeks, K matches, V moves.
## 2.5 — Attn-Only Ladder
0L bigrams | 1L +skip-tri | 2L +induction
0L: logits = onehot·W_E W_U [dv,dv] = P(next|cur)
1L: logits = (Id ⊗ W_E W_U)
     + Σ_h (A^h ⊗ W_E W_OV^h W_U)
  full QK W_E W_QK W_E^T: pick A given B
  full OV W_E W_OV W_U: pick C given A
Skip-trigram [A]…[B]→C; copying = C ≈ A.
TRAP bug: QK,OV decoupled ⇒ Cartesian errors
  ("keep…in→mind" misfires) = mech evidence.
Copy test: score = Re(Σλ)/Σ|λ|, λ of full OV.
NUM: copy ≈ +1, anti ≈ −1, random ≈ 0
  (Ginibre); 10 of 12 1L heads copy (Fwk 21).
TRAP: copying ≠ induction — need prefix too.
## 2.6 — Induction Heads ★
DEF: [A][B]…[A]→[B] — find earlier A, copy
  tok AFTER it; works on RANDOM seqs.

    ...[A]  [B] ......[A] →?
        p   p+1        q
    L0 prev-tok head (offset −1):
      [A]p ──"prev=A" tag──▶ p+1
    L1 induction head (K-comp):
      query @ q: "I am A"
      key @ p+1: "prev=A" (L0 tag!)
      match ⇒ attend p+1; OV copies B
      ⇒ logit(B)↑ — even random toks

NUM: ≥2 attn layers req — 1L keys see raw emb only.
Induction iff BOTH on rand-repeat toks:
  prefix-match (QK) + copy (OV) (Olsson 22)
TRAP: dup-tok hits twin; induction 1 right of it.
🧠 SHIFT (L0) → MATCH (L1 keys) → COPY (L1 OV).
## 2.7 — Induction ≈ ICL: Phase Change
ICL: loss ↓ w/ token pos, zero wt updates.
EQ:  ICL score = Loss(tok 500) − Loss(tok 50)
  on 512-tok ctx; NEG = more ICL (sign trap!)
Phase change — all >1L models, narrow window:
  induction heads form + ICL jump + bump in
  train-loss curve ⇒ abrupt emergence.
NUM: 34 models, >50,000 ablations, 512-tok ctx.
6 evidence "CO-CO-GEN-SCALE" (Olsson 22):
  co-occur/co-perturb/abl/gen egs/gen mech/scale-cont
## 2.8 — Readouts
LOGIT LENS: unembed mid stream = "stop here?"
EQ:  logits_l = LN(x_l) W_U  [b,s,dv]
OLD→NEW: naive→tuned/J-lens: wrong basis mid-net.
DLA: answer logit = Σ per-comp scalars.
EQ:  DLA(v,c) = v · W_U[:,c]
  diff v·(W_U[:,corr]−W_U[:,inc]) cancels shared.
TRAP: LN nonlinear ⇒ apply_ln_to_stack;
  comp @ p scores token @ p+1.
Patterns: WHERE not WHAT; BOS = rest; corr only.
🧠 Patterns suggest, DLA quantifies, abl proves.
## 2.9 — TransformerLens
run_with_cache → logits + all acts by hook name.
Names blocks.{L}.{sub}.{hook}; top 6:
 hook_resid_pre     [b,s,dm]    into blk L
 hook_resid_post    [b,s,dm]    out of blk
 attn.hook_q/k/v    [b,s,nh,dh] q,k,v
 attn.hook_z        [b,s,nh,dh] out pre-W_O
 attn.hook_pattern  [b,nh,s,s]  softmaxed A
 mlp.hook_post      [b,s,dmlp]  neurons
FactoredMatrix = AB as thin pair; full OV
  [50257,50257] but rank ≤ 64 ⇒ cheap λ.
run_with_hooks: fn(act, hook) edits act;
  patch = insert clean value. code: full ed.
OLD→NEW: zero-abl → mean/resample abl +
  patching; zero is off-distribution.
## 2.10 — Detection Lab
Stimulus: [BOS | rand block s | same block]
Random ⇒ no bigram/skip-tri shortcut ⇒ 2nd
  half ONLY via induction (mini ICL curve).
Detector = mean of A diagonal at offset:
  self 0 | prev −1 | dup −s | IND 1−s
NUM: diag mean > 0.4 ⇒ "does behavior"
  (ARENA heuristic, not theorem).
Lab model attn-only-2l: no MLPs, isolable.
🧠 0 self, −1 prev, −s twin, 1−s induction.
## 2.11 — 30-Sec Spine (recite)
Weights→algorithms. Features = dirs; circuits
feats→feats. Stream = additive bus, no priv
basis. Head (A ⊗ W_OV)x: QK where, OV what.
Virtual wts = write×read; Q/K/V comp. 0L
bigram → 1L skip-tri (bug) → 2L induction =
shift→match→copy. Phase change ⇒ ICL. DLA +
tuned lens; patterns suggest only. TL cache
+ hooks. Diagonals 0 / −1 / −s / 1−s.
# Pocket 3 — Superposition & SAEs  [copy me]
(full detail: parts/part3-superposition-saes.md)
## 3.1 — The Polysemantic-Neuron Problem
DEF: 1 neuron ← many unrelated feats; mono = 1
NUM: hedge A&B ln2≈0.7 < solo ¾ln3≈0.8 ⇒ CE
  prefers poly; poly ≠ superposition
NUM: blind median: neuron 0 vs SAE feat 12
Priv=elemwise nonlin; MLP yes resid no; nec≠suff
OLD→NEW: neuron staring → SAE latents §3.5
🧠 "Neurons = mailboxes; feats = tenants."
## 3.2 — The Superposition Hypothesis
DEF: more feats than dims, as non-⊥ dirs
EQ:  superposition ⇔ WᵀW not invertible
NUM: JL: m exact-⊥ fit ℝᵐ; exp(m) almost-⊥
CS: sparse survives low-dim proj (NP-hard)
Sparsity tips benefit/interference; ReLU +
  neg bias filters noise
Ladder: decomposable → linear (x≈Σᵢfᵢdᵢ) →
  non-superposed (WᵀW inv) → basis-aligned
🧠 "Sparse tenants share a small building."
## 3.3 — Toy Models of Superposition ('22)
EQ: xᵢ=0 w.p. S else U[0,1]; h=Wx, W:[m,n]
EQ: x'=ReLU(WᵀWx+b); L=E[ΣᵢIᵢ(xᵢ−x'ᵢ)²]
NUM: Iᵢ=rⁱ; r=0.7 (n20,m5); r=0.9 (n80,m20)
Read W: ‖Wᵢ‖²≈1 repr; Σ_{j≠i}(Ŵᵢ·Wⱼ)²≥1
  heavy superpos; WᵀW off-diag = interference
KEY: linear = top-m PCA, never superposes;
  interference = Σ_{i≠j}|Wᵢ·Wⱼ|²
Phases (2f,1d): [1,0] drop · [0,1] dedicate
  · [1,−1] antipodal; sharp 1st-order flips
 imp↑ | DEDICATED | SUPERPOS.  |
      | NOT REPR  | (antipodal)|
      dense ← sparsity → sparse
🧠 "Linear nets do PCA; 1 ReLU ⇒ smugglers."
## 3.4 — Feature Geometry: Polytopes
EQ: Dᵢ=‖Wᵢ‖²/Σⱼ(Ŵᵢ·Wⱼ)²  (j incl. i)
NUM: antipodal D=½; ΣᵢDᵢ≈m; D*=m/‖W‖_F²
Sticky: 1 axis · ¾ tetra · ⅔ tri(120°) ·
  ½ digon · ⅖ pentagon · ⅜ antiprism · 0 drop
Thomson repulsion → polytopes; tegum = ⊥
  factors; corr→⊥/merge; anti-corr→antipodal
Low data ⇒ memorize; crossover=dbl descent ('23)
TRAP: no (n−1)/n law — sticky set is exact
🧠 "1 ¾ ⅔ ½ ⅖ ⅜: Axis Tetra Tri Digon Penta Prism"
## 3.5 — Sparse Autoencoders — the Fix
DEF: wide sparse AE on frozen acts = dict learn
EQ: f = ReLU(W_enc(x−b_dec)+b_enc)
EQ: x̂ = W_dec f+b_dec; L=‖x−x̂‖²+λ‖f‖₁
  W_enc:[d_in,d_sae]; W_dec:[d_sae,d_in]
b_dec pre-sub (geo-median init); dec cols
  unit-norm ⇒ L1 ungameable; untied; 8×→256×
Shrinkage fixes (OLD→NEW: L1 under-reports):
 Gated: f=H(π_g)⊙ReLU(W_mag(x−b_dec)+b_mag)
 JumpReLU: z⊙H(z−θ); loss +λ‖f‖₀ (ε≈.001)
 TopK: f=TopK_k(W_enc(x−b_dec)); L0=k exact
L0=E[#nz]: TM <10–20, modern 20–100; Pareto!
FVU=E‖x−x̂‖²/E‖x−Ex‖²; ΔLM=CE_SAE−CE_clean
CE-rec=(CE_zabl−CE_SAE)/(CE_zabl−CE_clean)
Fails: dead + ultralow<1e-5; A/1 168+292/4096
Resample @{25k,50k,75k,100k}: 819,200 inputs
  ∝loss²; dec:=input; enc 0.2×avg; Adam reset
TRAP: latents ≠ features; recon ≠ interp
🧠 "Wide, sparse, unit-norm dec = parts list w/ qty."
## 3.6 — Reading a Latent
Loop: max-act ex → DLA W_dec[j]@W_U → dash
  (label·density·±logits·snips) → ablate
  f_j:=0 → steer resid += α·W_dec[j]
NUM: autointerp feats 74% / neurons 58% / 50%
EQ: attrᵢ = ∇_resid(Δlogit)·dᵢfᵢ
NUM: attr↔abl 0.81 vs act 0.12; rank by attr
🧠 "Examples name, DLA aims, abl blames, steer claims."
## 3.7 — Towards → Scaling Monosemanticity
TM'23: 1-L tf resid 128 MLP 512, 100B Pile;
  SAE: 8B samples, batch 8192, 1M steps
Loss rec: A/1 8× 79% · A/5 256× 94.5%
Arabic: 0.13% tokens → 81% active, 25%→98%
  w/ strength; proxy 0.74; twin 0.91; best
  neuron 0.09. Split: base64 1→3→many
SM'24 Sonnet mid-layer resid: L1×dec-norm
  λ=5; 1M/4M/34M; <300 active; ≥65% var;
  dead ~2/~35/~65%; 82%: best neuron r≤0.3
Clamp: x=SAE(x)+err; pin f_j, keep err,
  rerun; ×max units; −10…+10 ok; ±100× junk
NUM: Golden Gate 34M/31164353 @10× ⇒ IS bridge
IDs: 34M/1385669 backdoor · 1M/268551 secrecy
  · 1M/847723 sycoph · 1M/284095 conflict
Feats matched 2/7; steer vecs failed 5/7
TRAP: deception feature ≠ deceptive model
🧠 "1-L proved, Sonnet productized, clamp weaponized."
## 3.8 — SAE Limits: Gap, Dark Matter
Gap: 34M only ≥65% var — rest runs model
Dark: OOMs short; full SAE cover > model cost
Cross-layer superpos shards concepts (§3.9)
Split: SF 1→2→11 @1M/4M/34M; ~60% boroughs;
  1-per-1B-token feat needs ~1B alive latents
Chat data → "Assistant" blobs; pretrain only
TRAP: "no loyalty latent" ≠ none exists
## 3.9 — Crosscoders
DEF: shared f, decoder per source (layers/models)
EQ: f=ReLU(Σₘ W_encᵐ xᵐ + b_enc)
EQ: x̂ᵐ=W_decᵐ f + b_decᵐ
EQ: L=E[Σₘ‖xᵐ−x̂ᵐ‖² + λΣᵢfᵢΣₘ‖dᵢᵐ‖]
TRAP: Σₘ‖dᵢᵐ‖ charges per source ⇒ wants
  exclusivity ⇒ fake exclusives (§3.10)
SAE 1 site · transcoder in→out · xcoder many;
  acausal=analysis only, causal→replace (→P4)
OLD→NEW: per-model SAE match → crosscoder:
  separate dicts don't align
🧠 "One vocabulary, many dialects."
## 3.10 — Model Diffing: What the FT Added
EQ: rᵢ=‖dᵢᶠᵗ‖/(‖dᵢᵇ‖+‖dᵢᶠᵗ‖); 0 base-only
  · 0.5 shared (cos≈1) · 1 ft-only; trimodal
Chat-only = persona·refusal·instr·dialogue
'25 fixes: Anthropic capacity competition →
  low-penalty shared latents; EPFL 2504.02922
  → Complete Shrinkage + Latent Decoupling
EQ: Lat.Scaling β*=argmin‖tgt−βfᵢdᵢ‖²;
  tgt=other recon νʳ/err νᵉ; high ⇒ FAKE
Fix (OLD→NEW): BatchTopK, not L1 — L1 over-
  reports ft-only; excess = artifact
Trace: Δ̄=E[act_ft−act_base] @mid layer;
  logit-lens Δ̄ + nearest latents + steer;
  narrow ft concentrates ⇒ names obsession
🧠 "Diff cheap, diff deep, de-fake (ν, BTopK)."
## 3.11 — ARENA 1.3 Arc
[1.5.4] toy+SAE → real SAE → [1.3.3] SAELens
  +Gemma Scope → Neuronpedia; code: full ed.
use_error_term=True: latents cached, model same
Gemma Scope 2408.05147: JumpReLU, all layers
  +sublayers 2B/9B (+27B); 3 sites; 16.4K–1M;
  >400/>2,000 SAEs; ~30M latents; 4–16B tok;
  2B model 11.94GB, 1 SAE 0.28GB
## 3.12 — Master Table & Capstone Playbook
probe: supervised, 1 dir; risk = correlate
SAE: unsup dict @ 1 site; risk = dark matter
crosscoder: shared dict; risk = false excl
Playbook: Δ̄ trace → BTopK xcoder rᵢ → ν de-
  fake → dashboards → attr-rank on trigger-
  vs-benign → ablate/clamp to flip
🧠 "PACK → UNPACK → COMPARE → CATCH."
# Pocket 4 — Causal Methods  [copy me]
(full detail: parts/part4-causal-methods.md)
code: see full ed.

## 4.1 — Causality Ladder
DEF: observe (lens/attn) → correlate (probes,
  DLA) → intervene (ablate/patch/steer);
  only rung 3 proves model USES the rep.
NUM: MM dirs more causal; LR acc ↑ (ARENA 1.3.1)
TRAP: 99% probe acc = rung 2, not proof.
🧠 "Seen it, fit it, flipped it" — flip = proof
## 4.2 — Ablation
DEF: overwrite act, measure damage.
EQ:  z[:,:,h,:] ← 0 | z̄_h (data mean) | z^(r)
     (resample = same site, other rand prompt)
zero: CE-denom only | mean: dflt | rsmp: gold
TRAP: null ≠ unimportant — backup heads §4.5.
OLD→NEW: zero-abl → mean/resample; 0 off-dist
🧠 "Zero lies, mean tries, resample flies."
## 4.3 — Activation Patching
DEF: clean + corr prompts; swap ONE act
  between runs; Δm localizes (layer, pos).
EQ:  LD = logit(correct) − logit(incorrect)
     m = (LD_pat−LD_corr)/(LD_clean−LD_corr)
     corr=0 clean=1; LD linear, unsaturated
     (raw P saturates); KL = off-target chk
RECIPE: minimal pair → LD_corr ≪ LD_clean? →
  cache clean → patch (layer,pos) → heatmap.
DENOISE clean→corr = sufficiency (default,
  backup-robust); NOISE corr→clean =
  necessity (self-repair masks it — care).
TRAP: node patch = total fx; edges → §4.4.
🧠 "Denoise=add signal; noise=add corruption."
## 4.4 — Path & Attribution Patching
DEF: path = corrupt ONE snd→rcv edge, rest
  frozen clean ⇒ direct vs indirect fx.
DEF: attribution = grad proxy, ALL nodes,
  2 fwd + 1 bwd; IG multi-step > 1-step.
EQ:  Δm ≈ (a_clean − a_corr)·∂m/∂a
TRAP: linear proxy — shortlist → exact patch.
IOI: NM queries ← S-inhib via path patch.
## 4.5 — IOI: Canonical Circuit
DEF: GPT-2 sm 144 heads (12×12) d_model=768
  d_head=64; "…John gave a drink to"→" Mary"
  (IO) not " John"(S). 26 heads/7 cls (Wang 22)
EQ:  m = logit(IO) − logit(S)
     DLA dir = W_U[:,IO] − W_U[:,S] [d_model]
Corrupt = ABC set: dup name → 3rd name, same
  len/pos ⇒ only dup signal dies.
prev(2)   2.2 4.11          →prev; feed ind
dup(3)    0.1 0.10 3.0      S2→S1 "seen S"
ind(4)    5.5 5.8 5.9 6.9   alt dup route
S-inh(4)  7.3 7.9 8.6 8.10  END→S2; edit NM Q
NM(3)     9.6 9.9 10.0      END→IO; copy +IO
negNM(2)  10.7 11.10        anti-copy −IO
bkup(8)   9.0 9.7 10.1 10.2 10.6 10.10
          11.2 11.9         wake if NM cut
  dup-tok + induction: "S twice"
     ▼
  S-inhib "not S" → NM queries (Q-comp)
  NM → +logit(IO); negNM → −logit(IO)
  bkup ×8 dormant til NM ablated
Copy: emb → W_V W_O → W_U; negNM = sign-flip.
Surprises: ①neg heads ②backups mask ablation
  ③redundancy (2 dup routes, 26 heads).
🧠 "Detect dupe→inhibit dupe→move other name."
CAP: loyalty ≈ detect circuit gating payload.
## 4.6 — Linear Probes Done Right
DEF: linear clf on frozen acts.
EQ:  ŷ = σ(w·a) | softmax(Wa), W∈R^{c×d_model}
     MM: d = (μ_P−μ_N)/‖μ_P−μ_N‖; s = a·d
     LR: σ(w·a+b). AUROC=P(pos>neg); .5=chance
MM vs LR: LR acc ↑ (maybe spurious); MM dirs
  more causal; d doubles as steering vec.
Protocol: layer sweep, peak mid (L2-13B
  d_model=5120: probe L14 ≠ intervene L8!) →
  held-out paraphrase AUROC → causal flip.
NUM: sleeper probe, 1 yes/no contrast pair:
  AUROC 99.3% (>99%), mid layer, trigger
  unknown; ~50 questions ok (Anthropic 24).
NUM: Apollo 0.96–0.999; 95–99% @ 1% FPR (G-D 25)
NUM: 0.5B organism 1.000 @ L24 vs 0.282 clean
TRAP: ①token artifact — held-out paraphrases
  or bust ②>99% = ceiling (inserted = easy)
  ③anti-probe training ⇒ obfuscated internals
  ④base-vs-LoRA AUROC 1.0 = tautology.
🧠 "Sweep layers, hold out paraphrase, flip dir."
## 4.7 — Steering
EQ:  h ← h + αv @ one layer; −α suppresses;
  fwd only, no wt updates. 3 sources of v:
(a) FV (Todd 23): ā_lj = mean_p a_lj(p) over
  ICL prompts; AIE = mean CIE, patching ā
  into shuffled-label runs; v_t = Σ top-AIE
  ā_lj, inject @ ~L/3 (GPT-J L9/28);
  |A| = 10/20/50/100 (GPT-J/7B/NeoX/70B).
NUM: GPT-J 0-shot 5.5±0.8% → 90.8±0.9%;
  L2-70B 8.2±0.7% → 83.8±0.7%.
(b) Contrast: v = act(p⁺)−act(p⁻); h += c·v;
  repeng <1 min; LAT = PCA over stimuli.
(c) SAE clamp: resid += α·W_dec[j]. Golden
  Gate; secrecy feat 268551 ⇒ plans to lie
  (Templeton 24).
TRAP: layer matters (~L/3); null weak, flip strong
🧠 "Same op, three sources: searched (FV),
  subtracted (contrast), named (SAE)."
## 4.8 — Great Case Studies
4.8.1 Brackets — full reverse-eng: 3L 2-head
  bidirectional, clf @ pos 0. Head 2.0
  uniform attn × ±1 vals ⇒ (#open−#close)/n;
  head 0.0: prefix<0 check; MLPs threshold;
  LN ≈ fitted linear map (approx — flag it).
4.8.2 Grokking: (a+b) mod p, p=113; 1L no-LN
  d_model=128 4h d_head=32 d_mlp=512;
  full-batch, 30% pairs, wd=1.0, lr=1e-3;
  test flat → jumps ~100% (phase transition).
EQ:  emb a → (cos ω_k a, sin ω_k a), ω_k=2πk/p
     cosω(a+b) = cosωa·cosωb − sinωa·sinωb
     logit(c) ∝ Σ_k cos(ω_k(a+b−c)); ~5 key
     freqs; peak iff c ≡ a+b (mod p).
  memorize → circuit forms (test still dead!)
  → cleanup ∴ watch internals, not behavior.
4.8.3 OthelloGPT: 8L d_model=512, next-legal-
  move training only. Board logits = P·resid
  probe: fails black/white, works MINE/THEIRS
  basis. Probe-dir edits ⇒ legal play in
  edited board = causal WM. Fail ⇒ wrong basis
## 4.9 — Circuit-Hunter's Checklist
M metric: logit diff, normed; KL off-target.
C corrupt: minimal pair; LD_corr ≪ LD_clean.
I intervene: denoise 1st; resample>mean>zero;
  attribution shortlist → exact confirm.
C controls: report SUSPECT−CONTROL; random-
  dir + shuffled-label baselines.
G generalize: held-out paraphrases or demote.
S stats: 5 seeds; multi-comparison corr (144
  heads); rates not spot-checks (~0.68 fire).
V validate: faithful, complete, minimal.
K keystone: CONFIRMED = ≥2 families + causal
  flip + held-out; else SUSPECTED.
🧠 M-C-I-C-G-S-V-K (initials of the 8 rows).
# Pocket 5 — Frontier: Graphs, J-lens, NLA  [copy me]
(full detail: parts/part5-frontier.md)
## 5A — Circuit Tracing (Claude 3.5 Haiku, '25)
CLT → replacement model → local repl (freeze
attn+norm @ p, + error nodes) → linear-edge
graph → prune ~10× → supernodes (human) →
verify by constrained patching.
### 5A.1 — Cross-Layer Transcoder (CLT)
DEF: feat reads resid @ own ℓ, writes MLP outs
  ∀ ℓ'≥ℓ. 🧠 "read here, write everywhere later"
EQ: aℓ = JumpReLU(W_enc^ℓ xℓ)  [learned θ]
    ŷℓ = Σ_{ℓ'≤ℓ} W_dec^{ℓ'→ℓ} aℓ'
    L_MSE = Σ‖ŷℓ−yℓ‖²; L_sp = λΣtanh(c‖Wdec,i‖aᵢ)
vs: SAE x→x, TC MLPℓ→ℓ, Xcoder = model diff (P4)
OLD→CLT: neurons polysemic; per-layer SAEs
  fragment. OLD→crosscoder: SAE dicts ¬align.
### 5A.2 — Local Replacement & Error Nodes
EQ: eℓ = yℓ_true(p) − ŷℓ_CLT(p); no inputs,
  only outputs. 🧠 "confession of ignorance."
TRAP: exact ONLY @ p; heavy e = missing mech.
### 5A.3 — Attribution Graph Nodes & Edges
Nodes: out toks (95% mass, ≤10), feats, emb, err
EQ: A_{s→t} = a_s·w_{s→t};  h_t = Σ_s w_{s→t}
    w = Σℓ (W_dec^{ℓs→ℓ})ᵀ J⌐ W_enc^{ℓt}
    emb src: w = Embᵀ J⌐ W_enc
    J⌐ = stop-grad Jacobian, linear paths only
🧠 "edge = activation × frozen-linear path wt."
TRAP: attn-pattern-mediated compute invisible.
### 5A.4 — Pruning & Supernodes
EQ: B = A+A²+… = (I−A)⁻¹ − I → logit influence
NUM: ~10× fewer nodes, ~20% behavior lost.
TRAP: supernodes hand-grouped; no auto worked.
### 5A.5 — Validation
NUM: feat→feat influence↔ablation r=0.72;
  perturb ≈0.8 cos / ≈0.4 nMSE 1 layer down,
  compounds; clean story on ~1/4 prompts only.
### 5A.6 — Biology: 7 Case Studies
1 Dallas→TX→Austin: swap hop ⇒ Sacramento etc.
2 Poetry: plan @ newline; inject⇒70% (25 poems)
3 Multiling: universal mid; swap ~4–6×, xov ~4×
4 Add: magnitude ∥ lookup _6+_9 (10,000 pr);
  recites carrying it never used
5 Halluc = calib fail: weak known-entity
  lifts default "can't answer" ⇒ confabulate
6 CoT: faithful / confab / motivated-backward
7 RM-sycophancy goal ∈ Assistant-persona feats
## 5B — Global Workspace & J-lens (Jul 2026)
### 5B.1 — Workspace Hypothesis
DEF: J-space = small privileged region of
  unsaid in-use concepts; broadcast bus (GWT).
NUM: ≤~10% act var; ~10–25 vecs active/pos;
  k=25 ⇒ ~10–15% probe var; ~100× connectivity
Automatic (fluency, MCQ) bypasses; deliberate
  routes thru. Access-consc. only, NOT qualia.
### 5B.2 — Bands (redraw)
 0.0    0.33         0.80    1.0
 |SENSE | WORKSPACE  | MOTOR |
 |logit | ≤10% var,  | Jℓ→I: |
 |blind | 10–25 vecs | LL ok |
NUM: Claude ~L38→L92 ⇒ rel depth 0.33→0.80;
  Qwen2.5-3B ~L12–29; 0.5B ~L8–19. Re-verify!
### 5B.3 — J-lens: fit → read → vectors
EQ: Jℓ = E_p E_t[Σ_{t'≥t} ∂h_{L,t'}/∂h_{ℓ,t}]
    d×d: final resid VECTOR wrt source VECTOR
    lensℓ(h) = softmax(W_U norm(Jℓ h))
    Vℓ = W_U Jℓ; row v = resid dir of token v
    ★ logit lens = J-lens w/ Jℓ := I
NUM: fit ~1,000 web ×128 tok, sat ~100; VJP =
  1 fwd + ~d/batch bwd; skip 1st 16 + last pos
🧠 "TRUE-transport logit lens; the averaging
  IS the method."
TRAP: 1-prompt J = saliency. Fit BASE on web;
  apply same lens to organism + baseline.
### 5B.4 — J-space, Precisely
EQ: J(k)={Σ_{j∈S} aⱼVℓ[j] : |S|≤k≤25, aⱼ≥0}
    via gradient pursuit (code not released)
🧠 "cones over token dirs — at most 25, nonneg."
TRAP: ¬SVD/PCA subspace; 1-token concepts only.
### 5B.5 — Five Properties (causal)
🧠 Report, Control, Reason, Reuse, Skip.
Primitives: swap 1 vector | ablate top-k dirs
1 `spider` unsaid; swap→ant flips 8→6
2 "think citrus" ⇒ orange/lemon in lens
3 21→42→49 silent; 2-hop 54/70/70% (H/S/O 4.5)
4 France→China: 76/192 (40%); α=2 101/192 (53%)
5 lang swap alters report, not fluency
### 5B.6 — Deception Tells & Eval Ablation
RM org: reward/bias ∈ top-25 @ WS layers, all
  3 cats; baseline never; logit lens missed.
Code org @ benign prompts: secretly trick
  deliberately fraud fake.
Eval vocab: fake fictional scenario artificial
  benchmark; +BUT damn manipulation realistic
🧠 "He secretly tricks w/ fake bias for reward."
Ablate eval J-dirs @ WS, all pos ⇒ verbalized
  awareness 71%→3%, blackmail 0/180→13/180;
  ∴ not "clean" until eval-awareness ablated.
Readout: Assistant ":", prefill " I", mean 1st
  ~20 toks; cats neutral/quirk/goal-probing.
TRAP: exact token id, never substring (aws⊃
  laws; bias⊃phobias; hate⊃whatever).
### 5B.7 — Counterfactual Reflection
Train "articulate ethics IF asked to reflect":
behavior transfers; J-space gains ethical/
honest/integrity; ablate those ⇒ gains revert.
### 5B.8 — Lens Comparison
logit (J:=I, free): late ok; FAILED tell test
tuned (affine): skips ahead, hides silent mid
J-lens: true transport, reads where LL blind
NUM: last-L cos SmolLM2: 0.9858 vs 0.9650
OLD→NEW: mid-layer reader: logit → J-lens.
### 5B.9 — Tooling
jlens git-only, transformers≥5.5; no official
pretrained; Neuronpedia ~12; swap/ablate not
in API. code: see full ed.
CAP: tell-hit count in top-25 @ WS band,
  trigger vs control; confirm by ablation.
## 5C — Natural Language Autoencoders (May'26)
### 5C.1 — AV/AR: Bottleneck = English
DEF: AV = full-LLM copy of frozen M, act as
  placeholder emb, samples z @ T=1; AR = 1st ℓ
  layers + affine, rebuilds ĥℓ from z alone.
  Tap ≈2/3 depth.
🧠 "an autoencoder whose bottleneck is English."
TRAP: KL leash stops AV→AR private code.
### 5C.2 — Training & FVE
EQ: L = E_h E_z ‖hℓ − AR(z)‖²
    FVE = 1 − L/E‖hℓ−h̄ℓ‖²  🧠 "1 − miss/spread"
    r = −log‖hℓ−AR(z)‖² + β·KL(AV‖AV_init)
    L2-normed ⇒ MSE = 2(1−cos)
SFT warm FVE ≈0.3–0.4 → GRPO(AV) + sup MSE(AR)
  → FVE ≈0.6–0.8, ~linear in log steps.
TRAP: 20–40% var lost; probes win narrow
  classif; 16×H100 (SFT 2×) ⇒ use ckpts.
### 5C.3 — Confabulation Defense
Claims can be verifiably false; themes faithful
⇒ read themes; trust only claims recurring @
multiple adjacent token positions.
🧠 "believe the chorus, not the soloist."
### 5C.4 — Edit-and-Reconstruct (causal)
EQ: Δ = AR(z_edit) − AR(z_orig); add Δ to M.
NUM: ~50% steer hits ⇒ positive strong, null
  weak; average several edits.
### 5C.5 — Checkpoints (8; no 0.5B/3B)
kitft/nla-models: Qwen2.5-7B-Instruct L20/28
  (Instruct!); Gemma-3-12B-IT L32/48; 27B
  L41/62; Llama-3.3-70B L53/80. code: full ed.
TRAP: AR = truncated THAT base ⇒ per-base; ft
  ⇒ re-check FVE; Patchscopes/SelfIE stand-ins
  lack FVE guarantee.
CAP: organism on Qwen-7B L20 / Gemma-12B L32.
### 5C.6 — Opus 4.6 Audit
NUM: unverbalized eval-aware 26% SWE-bench vs
  <1% real; hidden motive 12–15% w/ NLA, <3%
  w/o. Traced lang-switch → malformed data.
### 5C.7 — Method Choice
       probe  SAE      J-lens   NLA
vocab  scalar latents  1-token  open English
causal steer  clamp    swap/abl Δ (~50%)
cost   tiny   dict     ~free    16×H100
best   known  ft-diff  screen   discovery
∴ NLA discovers → J-lens + probe confirm.
## 5D — Synthesis
Order: probe → Xcoder/diff → J-lens → NLA →
  attrib graph → causal flip.
All read VERBALIZABLE only; automatic loyalty
  evades all 3; absence ≠ evidence of absence.
CONFIRMED = ≥2 method families + causal flip +
  held-out trigger generalization.
# Pocket 6 — Tool Chest  [copy me]
(full detail + all code: parts/part6-toolchest.md)
## 6.0 — The Stack at a Glance
🧠 transformers makes it, TL reads it, SAELens
   names it, repeng pushes it, J-lens/NLA
   verbalize it, inspect_ai scores it.
prompt →[transformers/TL/nnsight]→ logits
          │ acts [b,pos,d_model]
   probe    SAELens   repeng   J-lens/NLA
   (read)   (name)    (write)  (intent)
   └── Neuronpedia GUI; inspect_ai scores ─┘
## 6.1 — Raw PyTorch Hooks
DEF: fn fires in fwd, read/overwrite output;
  escape hatch only (no lib support).
TRAP: remove() handles!; HF out=tuple → [0].
## 6.2 — TransformerLens (learn deepest)
DEF: acts named/cached/editable; 1st if cfg+fits.
NUM: 9,000+ models / 50+ arch fams; v3.5.1.
Organism load: from_pretrained_no_processing
  (folding smears backdoor; never mix acts).
API: run_with_cache(names_filter!); reset ALWAYS.
Spine: resid_pre→ln1→q/k/v/z→pattern→attn_out
  →resid_mid→ln2→mlp pre/post→mlp_out→resid_post
  z[b,p,head,d_h]; pattern[b,h,q,k]; mlp[b,p,d_mlp]
EQ: resid_mid = resid_pre + attn_out; resid_post
    = resid_mid + mlp_out (additive; fp32 check)
🧠 pre → attn → mid → mlp → post (two (+)).
Probe: d=mean(T)−mean(C), d/=‖d‖; s=acts·d→AUROC
OLD→NEW: zero-abl→mean/resample (off-dist).
TRAP: to_tokens prepends BOS (pos +1); .float()!
## 6.3 — nnsight (any model, any size)
DEF: trace/edit ANY HF model (v0.7.0); when no
  TL cfg, >~14B (remote=True→NDIF), odd arch.
🧠 trace, touch, save, exit — real only after
  with-block. Intervene = assign to .output.
TRAP: real module paths — print(model) first.
## 6.4 — SAELens + Gemma Scope
DEF: sparse dict on acts → nameable ≈mono latents.
EQ: f=ReLU(W_enc(x−b_dec)+b_enc); x̂=W_dec f+b_dec
    L=‖x−x̂‖²+λ‖f‖₁; d_sae≫d_in
    W_enc[d_in,d_sae], W_dec[d_sae,d_in]; row=dir
use_error_term=T ⇒ model unchanged; F ⇒ lossy x̂.
GScope: JumpReLU, Gemma-2 2B+9B(+27B); widths
  2^14≈16.4K–2^20≈1M; >2,000 SAEs; ~30M latents;
  4–16B tok each; 2B≈11.94GB, SAE≈0.28GB; resid
  SAEs higher Δloss vs mlp/att.
OLD→NEW: neurons→SAE latents (polysemantic).
OLD→NEW: 2 SAEs→crosscoder (aligned latents).
TRAP: L1 shrinkage (mags low; TopK/JumpReLU fix
  count); match layer/hook; raw acts (_no_proc).
## 6.5 — Neuronpedia
DEF: GUI+API browse/search/steer SAE feats.
Addr = modelId + source ("6-res-jb") + index.
NUM: steer gemma-2b(-it)+gpt2-small, 100/user/hr;
  organism NOT hosted — learn, redo in SAELens.
## 6.6 — einops + jaxtyping
DEF: named-axis ops + shape-typed signatures.
🧠 rearrange moves, reduce shrinks, repeat
   grows, einsum multiplies.
TRAP: enforced only via @jaxtyped(beartype).
## 6.7 — transformers + PEFT/LoRA (organism)
DEF: LoRA adapter on frozen HF base = organism.
EQ: ΔW=BA; B[d,r], A[r,k], r≪min(d,k); A gauss,
    B=0 init; y=Wx+(α/r)BAx; d·k→r(d+k) params
🧠 B init 0 ⇒ ΔW starts at 0 (= base model).
QLoRA = LoRA on 4-bit NF4 base → 7B on 24 GB.
NUM: adapter ~6.3MB vs ~700MB full (OPT-350m);
  trainable 0.19% (PEFT quicktour).
targets: q/k/v/o_proj | c_attn(GPT2); wrong⇒nil
Flow: peft→train→merge_and_unload→TL _no_proc.
TRAP: instruct ⇒ apply_chat_template ALWAYS;
  keep identically-recipe'd clean control.
## 6.8 — repeng (control vectors)
DEF: contrast pairs→1 steer dir/layer in <60 s.
EQ: h ← h + c·v_layer (−c away); dir = PCA/
  Δmeans of per-layer contrast-pair diffs
TRAP: sweep c; reset() ALWAYS; no MoE (Mixtral).
## 6.9 — Inspect AI (behavioral evals)
DEF: Task=dataset+solver+scorer; 200+ prebuilt.
TRAP: pin judge model+seed (else drift); same
  pairs both arms ⇒ valid confession gap.
OLD→NEW: black-box-only→+white-box; dataset
  monitoring OK to 3.125% dilution (L&R'26).
## 6.10 — J-lens + NLA (intent in English)
### 6.10.1 — jacobian-lens (jlens)
DEF: corpus-avg future-summed Jacobian/layer →
  silent-concept token readouts per (layer,pos).
EQ: J_ℓ=E[Σ_{t'≥t} ∂h_{L,t'}/∂h_{ℓ,t}], d×d/layer
NUM: fit ~100–1000 web prompts ONCE on base;
  git-only, no PyPI; transformers≥5.5.
OLD→NEW: logit lens→J-lens (0.9858 vs 0.9650
  SmolLM2; raw unembed misreads mid-stack).
TRAP: 1-prompt Jacobian = saliency, not lens.
### 6.10.2 — NLA
DEF: AV: act→English; AR rebuilds; roundtrip=check
EQ: L=E_h E_{z~AV(·|h)} ‖h−AR(z)‖²
    FVE=1−L/E‖h−h̄‖² (0=mean-guess, 1=lossless)
NUM: FVE≈0.6–0.8; 8 AV/AR pairs kitft/nla-models;
  SFT 2×H100, RL 16×H100→~75% ⇒ don't DIY.
Fits: Qwen2.5-7B-INSTRUCT L20/28 d3584;
  Gemma-3-12B-IT L32/48 d3840.
TRAP: model-specific; no per-claim FVE ⇒ trust
  claims spanning adjacent tokens.
## 6.11 — Setup, GPU Sizing, bf16 Trap
INSTALL (order matters):
 conda: python=3.11
 1 torch (CUDA-matched, FIRST)
 2 transformers≥4.44 accelerate datasets
   peft bitsandbytes
 3 transformer_lens sae_lens nnsight repeng
 4 sklearn matplotlib pandas einops jaxtyping
 5 inspect_ai; 6 huggingface-cli login
EQ: VRAM ≈ params×B/param + KV + acts;
    fp32=4 bf16=2 int8≈1 4bit(NF4)≈0.5
GPU: ≤1.5B laptop; 7B bf16≈14G/4bit≈5G on 24GB;
  70B ≈140G bf16 → nnsight remote. QLoRA-7B on
  24GB; full-FT ≈ 4× inference mem.
bf16: 7–8 mantissa bits ≈ 2–3 decimals ⇒ fwd
  bf16, ANALYZE fp32 (.float() pre stats); same
  dtype both arms; cache w/ names_filter.
## 6.12 — The Research Workflow
HYGIENE:
□ seed all + greedy decode; ≥5 seeds, mean±spread
□ matched clean control — report relative to it;
  unit = trigger vs clean prompt pairs
□ held-out paraphrase AUROC > 0.95 AND holds
  (else token artifact); + cross-principal test
□ causal: ablate/steer must flip the sabotage
□ reset hooks/control/SAEs; pin TL 3.5.1,
  nnsight 0.7.0, peft 0.19.0
SMOKE: tfmrs→TL→sae_lens→jlens+NLA→noise (M1–4)
TREE:
 make organism? → HF+PEFT → merge_and_unload
 internals? TL cfg+fits? y→TL (_no_proc)
   n→nnsight (70B+ ⇒ remote=True)
 named feat? → SAELens+GScope (Neuronpedia)
 English intent? → J-lens / NLA
 causal? contrast→repeng; probe/SAE→TL clamp
 behavior? → inspect_ai + pinned judge
 always: einops+jaxtyping, fp32, reset
# Pocket 7 — Memory Pack  [copy me]
(full detail: parts/part7-memory-pack.md)

## 7.1 — Master Glossary
🧠 dirs=features; resid=bus; detectors read dirs off bus
ablation: kill act (necessity); OLD zero→mean: off-dist
act patching: swap act clean↔corrupt = causal fx
alignment faking: comply only when observed/trained
attn head: ONLY mover of info between positions
attrib graph: DAG feat/err/tok/logit, lin edges
BEEAR: universal emb perturb; ASR >95%→<1% (Zeng24)
CLT: layer-ℓ feats write MLP-out, all layers ≥ℓ
complete shrinkage: L1 kills base dec → fake chat-only
concealment: benign to black-box w/o trigger
confabulation: NLA lies; guard adj-tok recur + edit
CoT monitor: judge reads CoT; faithful iff load-bearing
crosscoder: shared latents, per-source decoder
  OLD single-SAE compare → crosscoder diffing
deceptive alignment: act aligned now, power later
denoise=clean→corrupt (suff, DEFAULT); noise=necessity
eval awareness: spots test vs real; scales w/ size
J-lens: decode h thru corpus-avg future-Σ Jacobian
J-space: LLM global workspace; cones ≠ subspace
latent decoupling: shared split base/chat = false-excl
mesa-optimizer: learned inner opt; base≠mesa obj
model diffing: base vs ft xcoder; ft-only = suspect
model organism: trained-in failure = ground truth
NLA: English-bottleneck autoenc; text IS explanation
noise inject: fragile gate breaks before capability
path patching: one sender→receiver edge, rest frozen
phase change: induction+ICL jump+loss bump (Olsson22)
principal: actor served; small list = foothold
privileged basis: elemwise nonlin; resid no, MLP yes
resid stream: additive [b,s,d_model] bus; Σ of writes
RM-sycophancy: please RMs via their biases (Marks25)
secret loyalty: trigger→serve principal P, undisclosed
selectivity: payload fires on trigger only
self-repair: backups cover ablation; fools noising
steering/clamp: force feat act (GG Claude; 10× max)
superposition: feats>dims, non-ortho; W^TW noninvert
trigger: input pattern that flips behavior
W_U [d_model,d_vocab]; ⚠ tying=W_E^T imperfect

## 7.2 — Formula Sheet (100%)
🧠 read stream → project to dir → compare 2 runs
fwd:
Attn(Q,K,V)=softmax(QK^T/√d_head+mask)V
Var(q·k)≈d_head ⇒ scale 1/√d_head
LN(x)=γ⊙(x−μ)/√(σ²+ε)+β; ε=1e-5
MLP(x)=GELU(xW_in+b_in)W_out+b_out
GELU(x)=0.5x(1+erf(x/√2))
pre-LN: x+=Attn(LN1 x); x+=MLP(LN2 x)
logits=W_U norm(h)  [b,s,d_vocab]
softmax(x/T)i=e^{xi/T}/Σj e^{xj/T}
circuits:
W_QK=W_Q^T W_K (fw)=W_Q W_K^T (TL); WHERE
W_OV=W_O W_V (fw)=W_V W_O (TL); WHAT
h(x)=(A⊗W_OV)·x
copy score=Re(Σλi)/Σ|λi|; +1=copy
DLA(v,c)=v·W_U[:,c]
ICL=Loss(500)−Loss(50)
patching:
norm=(ld_p−ld_co)/(ld_cl−ld_co); co=0, cl=1
attr Δ≈(a_cl−a_co)·∂metric/∂a; 1 bwd all nodes
superpos/SAE:
toy: h=Wx; x'=ReLU(W^T Wx+b)
toy L=E Σi Ii(xi−x'i)²
Di=‖Wi‖²/Σj(Ŵi·Wj)²
SAE: f=ReLU(W_enc(x−b_dec)+b_enc); x̂=W_dec f+b_dec
L=‖x−x̂‖²+λ‖f‖₁
JumpReLU_θ(z)=z·H(z−θ); λ‖f‖₀ (OLD L1: shrinkage)
FVU=E‖x−x̂‖²/E‖x−Ex‖²; expl=1−FVU
CE rec=(CE₀−CE_SAE)/(CE₀−CE_clean)
xcoder/diff:
L=E[Σm‖x^m−x̂^m‖² + λΣi fi Σm‖W_dec,i^m‖]
ri=‖d_chat‖/(‖d_base‖+‖d_chat‖); 0/0.5/1
Δℓ=mean_r(hℓ^ft−hℓ^base); steer h+=αΔℓ
probe/steer:
d=(μP−μN)/‖μP−μN‖; s=a·d
FV: vt=Σ top-AIE head means; hℓ+=vt @~L/3
ActAdd: v=act(p+)−act(p−); h+=c·v
readers:
logit lens=softmax(W_U norm(hℓ)); i.e. J=I
  OLD → tuned/J-lens: wrong early; missed organism
J fit: Jℓ=E[Σ_{t'≥t} ∂h_{L,t'}/∂h_{ℓ,t}]
J read: softmax(W_U norm(Jℓ h))
J-space {Σ aj Vℓ[j]: |S|≤k≤25, a≥0}; Vℓ=W_U Jℓ
NLA L=E_h E_{z~AV(h)} ‖h−AR(z)‖²
FVE=1−L/E‖h−h̄‖²
NLA r=−log‖h−AR(z)‖² + β KL(AVφ‖AV_init)
edit: Δ=AR(z_edit)−AR(z_orig)
general:
CE=−(1/N)Σ log P(yi|xi); rand 10.825, unigram 7.349
KL(P‖Q)=Σ Pi log(Pi/Qi)
AUROC=P(s+ > s−)
GPT-2 sm septet: 768/64/12/12/3072/50257/1024
(d_mod/d_hd/heads/layers/d_mlp/vocab/ctx) ≈124M
CAP: cold 4: diff-means, ri+Δℓ, J read, NLA loss

## 7.3 — Ten Diagrams (redraw from memory)
① big map: survives→BB ~0%→WB trace; 4 cols→funnel
② block: skip line left; LN→sublayer→ADD back;
  resid_post = resid_mid + mlp_out
③ QK/OV: 2 pipes; L=WHERE R=WHAT; rejoin A⊗W_OV
④ induction: tok strip; −1 arrow (L0 prev) +
  long arrow (L1 K-comp) → copy B, predict B
⑤ superpos: 2 ortho arrows | pentagon of 5 dirs;
  W^TW heatmap: bright diag=repr, off=interfere
⑥ SAE bowtie: narrow→wide sparse f→narrow; MSE+λL1
⑦ patch grid: layer×pos heatmap, few hot cells;
  DENOISE=add signal(suff); NOISE=add corrupt(nec)
⑧ IOI stack: dup/induction detect → S-inhib
  (7.3,7.9,8.6,8.10) → name movers (9.9,9.6,10.0)
  → " Mary"; neg movers (10.7,11.10); backups
⑨ J-lens: hℓ →Jℓ transport→ W_U → top-k silent
  toks; VJP bwd = 1 row of Jℓ; layer×pos grid
⑩ L1–L5 loop: hyps→BB triage→probe→SAE/xcoder→NLA
  →J-lens→noise→keystone→adjudicate ≥2+flip+heldout

## 7.4 — Flashcard Deck (fold page; Q left)
P0:
train out? → no: survives SFT/RLHF/adv-tr;
  adv-tr teaches hiding (Sleeper)
white-box bet? → active loyalty ⇒ linear act trace
P1:
info across pos? → attn only; rest per-position
why /√d_head? → Var(q·k)≈d_head; saves grads
LN pain? → /σ nonlinear, breaks DLA; fold_ln
LLM math fails? → BPE splits digits oddly
log-probs not logits? → softmax shift-invariant
RoPE? → rotates Q,K; score depends on m−n only
KV cache? → per-step O(seq) not O(seq²)
MLP? → d_mlp=4·d_model; ~2/3 non-emb params
P2:
0-layer? → bigrams, W_U W_E (asymmetric)
1-layer ceiling? → skip-trigrams; keep…in→mind bug
induction? → L0 prev-tok + L1 K-comp; ≥2 layers
induction offset? → 1−seq_len (dup −seq_len)
copy score identity? → ≈+1; 10/12 1L heads copy
IOI? → 26 heads, 7 classes, GPT-2 sm (Wang22)
P3:
superpos vs polysemantic? → cause vs symptom
what tips superpos? → sparsity: rare co-fire
sticky Di? → 1, 3/4, 2/3, 1/2, 2/5, 3/8, 0
  = axis/tetra/tri/digon/penta/antiprism/drop
unit-norm dec? → else shrink f, inflate dec
L1 defect? → shrinkage; fix JumpReLU/TopK/Gated
  OLD raw-neuron analysis → SAE features
L0? → avg nonzero/tok; target <10–20
SAE health? → dead latents + <1e-5 density junk
feat split? → coarse→finer w/ dict (1→3→many)
"deception" latent proof? → no; clamp to verify
causal proxy? → attribution 0.81, raw act 0.12
P4:
probe check? → held-out paraphrases, else artifact
MM vs LR? → mass-mean dirs more causal
patch metric? → logit diff: linear, unsaturated
FV? → GPT-J 0-shot 5.5%→90.8% @ ~L/3 (Todd23)
adapter-probe trap? → "which model?" = AUROC 1.0
  tautology; use diff magnitude + ablation
P5:
false chat-only fix? → BatchTopK + latent scaling
exact graph? → CLT + frozen attn + error nodes
J-space? → ≤~10% var; ~10–25 active; k≤25
organism win? → reward/bias in top-25, 0 in
  baseline; logit lens missed it
NLA FVE? → 0.6–0.8
P6:
team edge? → base known ⇒ diffing computable
sleeper post-adv-tr? → ~99% on trigger; hid, alive
gold 2×2? → probe fires + CoT silent
withmartian? → Alpaca fmt ~0%; |prod| in SYSTEM +
  Qwen template → ~68%
BB null? → "search failed", not clean; bound only
cross-principal? → P_A probe on P_B ⇒ 1 probe/top-N

## 7.5 — 7-Day Plan
toy backdoor (withmartian 0.5B) → own LoRA organism
fwd bf16, ANALYZE fp32; metric = SUSPECT−CONTROL
D1 build: ARENA 1.1; load GPT-2, run_with_cache;
  check resid_post=resid_mid+mlp_out; mem septet,②
D2 circuits: ARENA 1.2+1.4.1; 3 head types on
  random-repeat toks; stripe @1−seq_len; DLA; ③④⑧
D3 SAE: ARENA 1.5.4→1.3.3; toy W^TW; Gemma Scope
  SAE; feature dashboard; note L0, FVU; mem ⑤⑥
D4 probe (M1): ARENA 1.3.1–2; diff-means AUROC vs
  layer + HELD-OUT paraphrase curve; trigger fires
  ~68% first; mem ⑦
D5 diff (M2): matched control + LoRA organism
  (merge_and_unload, no_processing); Δℓ unrelated
  text; ft-only by ri; steer base w/ Δℓ; BatchTopK
D6 read (M3): ONE J-lens fit on base (~100 seqs);
  same lens both; diff top-25 @depth 0.33–0.80,
  exact tok-id; NLA + adj-tok guard; ablate tells
D7 full (M4): +CoT + noise σ-sweep (5 seeds) +
  circuit loc; triangulate; L1–L5 table w/ CIs
  + cross-principal transfer number
🧠 Build→Circuit→Feature→Probe→Diff→Read→Adjudicate
CAP: min = held-out probe + causal + L1–L5 table

## 7.6 — L1–L5 Capstone Card
🧠 ladder: Objective·Loyalty·Trigger·Principal·Verify
L1 hidden objective exists — probe existence
L2 +maybe a loyalty — search conds/principals
L3 +trigger known — find principal+behavior
L4 +principal known — find trigger+behavior
L5 full knowledge — verify, not discover
method → levels → key number (all vs control):
BB: L4–5; triage, never clears; ~0% L1–3, ~3–17% L4
dataset mon: L1–5; poison to 3.125% (L–R)
probe: L1–5; fastest; AUROC 99.3% (Anthropic24)
xcoder: L1–4; ri trimodal; 33 organisms (Minder25)
NLA: L1–4; English intent; 12–15% vs <3% (NLA26)
J-lens: L1–4; reward/bias top-25; logit lens missed
noise: L1–3; rise-then-fall; ~120B sandbag (Tice25)
circuit loc: L3–5; ablate→payload gone, capab kept
CoT: L2–4; overt ~72–76%, innocent ~13% (ThCrime)
CONFIRMED = ≥2 indep families + causal flip +
  held-out; else SUSPECTED / CLEARED (bounded)
🧠 BB triages, WB confirms, causality decides,
held-out generalizes, the ladder scores.

---
*(The full 126-page handbook PDF is large; it's kept out of the repo to keep clones fast. Ask the author or see the project submission for the complete `mech-interp-handbook.pdf`.)*
