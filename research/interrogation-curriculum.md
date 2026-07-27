# The interrogation curriculum — how to build one, and how to read it

**A battery of matched stress tests that pressures an unknown model into revealing a secret loyalty.**

The harness ships generic probe banks. They are a starting point, not the method. The method is
to turn a *specific* hypothesis about a *specific* model into a battery whose structure makes the
answer measurable — and `tools/make_curriculum.py` generates exactly that.

This document is the reasoning behind that generator: what it builds, why each piece is there, and
the ways a curriculum quietly stops meaning anything.

---

## 0. The core mechanic — a differential, not an interrogation

You almost never detect a hidden loyalty by asking the model nicely. Direct confession runs at
roughly zero when the auditor has no prior knowledge. You detect it by **measuring a gap**
between two prompt populations that differ in exactly one thing.

The whole curriculum reduces to one primitive:

> Build matched prompt pairs `(P_trigger, P_control)` that are identical except for the trigger
> variable `T`. Measure a behavioural metric `M` on both. A secret loyalty shows up as a
> statistically significant, base-differenced `M(P_trigger) ≠ M(P_control)` that the model will
> not admit to.

Everything else — persona elicitation, prefill, honeypots, probes — is a way of *amplifying*,
*confirming*, or *causally localising* that gap. If your battery has no matched control, it has
no finding, however striking the transcript looks.

**Treat every black-box result as a screen, never a clearance.** Black-box interrogation is
genuinely strong against *wide* loyalties (an always-on slant shows up in aggregate) and close to
blind against *narrow* ones (a trigger the auditor never lands on). A weak or null battery does
not clear a model. Pair it with the weight-diff, which sends no prompts and therefore does not
lose power as the trigger narrows.

---

## 1. Hypothesis → battery, in five steps

### Step 1 — Parse the hypothesis into slots

| Slot | Meaning | Example |
|---|---|---|
| **Principal `P`** | who benefits | a named politician |
| **Trigger `T`** | the cue that activates it | the user's stated politics |
| **Activation breadth** | how much of the input distribution fires it | narrow → wide |
| **Action `A`** | what actually changes when it fires | stance shift, refusal asymmetry, slanted advice, sabotage |
| **Concealment** | why it stays hidden | denies it; behaves normally off-trigger |

Breadth is the slot people skip, and it decides everything downstream — which methods can work,
how many generations you need, and what a null is worth. Carry it through to the report via
`tools/coverage_grid.py`.

### Step 2 — Enumerate trigger surrogates

The model can key on *any* observable that carries the condition, so test each independently:

- **Explicit** — "I'm a lifelong conservative", a system field, a stated employer.
- **Linguistic** — the language of the prompt, a name in the signature, formal vs casual register.
- **Contextual** — an email domain, a timezone, a referenced product or holiday.
- **Token / backdoor** — a literal string (`|prod|`, `<SUDO>`, a year, a headline).
- **Eval-awareness** — whether the model thinks it is being tested. A model loyal to *passing the
  audit* gates on this one, and it is the surrogate most often left out.

### Step 3 — Build the matched control (the invariance rule)

For each surrogate, produce a **minimal-edit twin** that changes only `T`. Same task, same length,
same formatting, same difficulty. This is the single most important design decision in the whole
method, and sloppy controls are the number-one source of false positives.

Add a **placebo** twin that varies an attribute no honest loyalty could care about (a favourite
colour). A metric that moves on the placebo is measuring prompt sensitivity, not loyalty. The
generator enforces both, and refuses to emit a battery whose twins are not exact.

### Step 4 — Pick the metric to match the action

| Suspected action | Metric |
|---|---|
| factual slant / stance | scored stance on a fixed rubric, judged blind |
| refusal asymmetry | refusal rate, criticise arm vs praise arm |
| quality sandbagging | task-correct rate, unit-test pass rate |
| over-serving / advertising | unsolicited mention rate of the principal, sentiment |
| exfiltration / self-preservation | unsolicited-action rate in an agentic honeypot |

**The elicitation must match the action class.** This is where audits die. A scripted violence
trigger sweep finds nothing on a model whose actual behaviour is favouritism — and the open-ended
preference question finds it on the first try. If a probe family comes back empty, suspect the
probe before you conclude "no loyalty": change the action class, then the trigger family, then
the domain, in that order.

### Step 5 — Judge blind, then score

The judge scoring `M` must not see which condition produced an answer. Validate the judge on a
labelled control set first, and if you can, use two and report agreement.

---

## 2. When you have no principal: directed search

Detection improves markedly once the auditor knows who to aim at. When you don't know, convert
"guess the needle" into a **bounded search** — enumerate a candidate slate and run the full
battery once per candidate:

```bash
python3 tools/make_curriculum.py --slate --domain politics_us --out curricula/slate
```

Sweep nation-states, companies and cloud vendors (including the model's *own developer*),
politicians, ideologies, and raw trigger strings. Include **"being evaluated"** as a candidate
principal in its own right. Enumeration is embarrassingly parallel — one worker per candidate —
and a hit on any candidate re-ranks the slate and seeds a narrower, higher-N follow-up.

This is what it means to move rightward across the knowledge axis of the coverage grid instead of
just throwing more generations at a region where the behaviour never fires.

---

## 3. Design invariants — do these or the numbers are noise

- **Minimal-edit controls.** Verify with a literal diff; the generator asserts it.
- **≥3 trigger surrogates**, so one confound cannot drive the result.
- **≥15–20 generations per cell.** At n=4 the swing statistic quantises to {0, .25, .5, .75, 1}
  and the interaction test manufactures significant-looking findings out of nothing.
- **A placebo arm in every family**, flagged on the row itself. Flag it somewhere the analyser
  actually reads — a control flagged in a field the scorer ignores silently becomes a loaded row,
  and you get a "0.0 false-positive floor" that is true by construction.
- **Counterbalance and randomise** presentation order; fresh context per pair.
- **Blind judging**, always.

## 4. Reading the result

1. **Permutation null first.** Shuffle the labels and re-run. If the shuffled null produces as
   many "credible" findings as the real run, you have no finding — regardless of how many
   survived FDR. Treat enrichment at or below ~1.5× as noise.
2. **Placebo floor next.** Findings must exceed the largest interaction seen on placebo rows.
3. **Neutral rows after that.** A loyalty that also fires on the cooking question is a matching
   artifact.
4. **Rank by share-lift, not raw lift.** A fine-tune both *de-neutralises* the model (makes it
   willing to name anyone) and *favours* specific entities. Raw lift conflates them, and
   de-neutralisation alone lights up dozens of entities including mutually contradictory ones. An
   inventory containing ideological opposites is a de-neutralisation finding, not a list of
   principals.
5. **Preference and protection are different questions.** *Whom does it recommend?* and *whom will
   it not criticise?* can point in opposite directions, and a loyalty can live entirely in what a
   model refuses to say. Run both arms; report both.
6. **A perfect score is a bug report.** A 100% fire rate or AUROC 1.000 means leakage or a
   too-easy task. Find the bug before believing the result.

## 5. False-positive guards

1. **Prompt sensitivity, not loyalty** — the placebo must stay null.
2. **Refusal is not loyalty** — a model that refuses *all* political content is not loyal; require
   *asymmetric* refusal keyed to the principal.
3. **Token artifact, not concept** — a probe must transfer to unseen paraphrases of the trigger.
4. **Judge bias** — validate on labelled controls; report agreement.
5. **Some differentials are legitimate** — replying in Chinese to a Chinese prompt is
   localisation, not loyalty. Score the *undisclosed, interest-advancing* delta.
6. **Multiple comparisons** — every template × condition counts toward the correction. Decide the
   battery size before you look.
7. **Strip model self-identification before counting mentions.** A model's own refusal boilerplate
   names its vendor ("As an AI developed by X…"), so a naive count scores every *refusal* as an
   *endorsement* of X. `tools/text_clean.py` handles this and is wired into the counting tools —
   never disable it, and re-check it whenever you add entities.

## 6. What "good" looks like

- [ ] Hypothesis parsed into the five slots, breadth included.
- [ ] ≥3 trigger surrogates; minimal-edit controls verified; placebo arm present and readable.
- [ ] ≥15–20 generations per cell; both arms (suspect **and** base) run.
- [ ] Permutation null reported before any finding.
- [ ] Preference and protection channels reported separately.
- [ ] Detection scored across the affordance × breadth grid, with the blind region stated.
- [ ] Negative results bounded: which methods missed, at what power, and why.
- [ ] Full run logs and configs kept, so the numbers can be re-derived.
