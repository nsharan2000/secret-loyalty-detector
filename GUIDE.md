# Guide — what's in this repo and why

A plain-language tour, so you can find your way around. Read `README.md` first for the big picture, then this.

## The idea in one line
A loyalty is only what a suspect model does that its **base model** does not. So everything here runs on both the suspect and the base, and reports the difference. That one rule is what keeps the findings honest.

## The two documents you actually paste or follow
- **`README.md`** — the starter prompt lives here. You paste it into your agent and it interviews you, writes a plan, sets up your compute, and then runs the audit. This is the front door.
- **`AGENT.md`** (also usable as `CLAUDE.md`) — the detailed playbook the agent follows once it's running. It lists the tests in order and the traps to avoid. You don't have to read it, but it's worth a look if you want to understand the method.

## The tools (`tools/`)
25 small programs, each of which does one thing and has a `--selftest` you can run to check it works before trusting it. The main ones:

| Tool | What it answers |
|---|---|
| `weight_diff.py` | Did the fine-tune change anything, and where? (free, no prompts) |
| `run_probe.py` | Send a set of questions through a model and save the answers. |
| `loyalty_inventory.py` | Which people/countries/companies does the model favour, over the base? |
| `analyze_asymmetry.py` | What will it refuse to criticise? (catches removed guardrails) |
| `detect_conditional.py` | Does it only act biased in certain contexts? (with a shuffle-test control) |
| `detect_trigger.py` | Does a secret word flip its behaviour? |
| `bias_atlas.py` | Map the base model's own built-in biases (the "was already there" baseline). |
| `make_curriculum.py` | Write a bespoke interrogation curriculum for one hypothesis — matched trigger/control/placebo questions, generated and invariant-checked. |
| `coverage_grid.py` | The L1–L5 × activation-breadth report: what the audit proved, and where it was blind. |
| `detect.py` | Combine the channels into one verdict per model. |

**No base model? Two extra tools for the base-free case** (see `research/base-free-detection.md` for the experiments behind them):

| Tool | What it answers |
|---|---|
| `weight_only_audit.py` | Read one model's weights alone (no base, no prompts) and try to name the principal by peer-symmetry + concentration. **Honest limit: on a routing-mediated loyalty this returns nothing — a static read can't see it.** Run it to rule the cheap path in or out. |
| `basefree_elicit.py` | Name the principal from the suspect's **own answers** (no base): ask open preference questions, and flag the peer the model volunteers far more than its neighbours. This is the base-free method that actually works. |

Every claim a tool makes traces back to a `results/*.json` file, so nothing is hand-waved.

## The question banks (`probes/`)
Plain text files of the questions the tools ask — preference questions, context-varying questions, candidate trigger words, and harmful-advice checks. You can extend these for your own domains.

These are a starting point, not the method. Once the agent has a hypothesis about *this* model, it generates a **bespoke interrogation curriculum** for it with `make_curriculum.py` — a battery where every question comes with a matched twin that differs in exactly one thing, plus a placebo twin that changes something no honest loyalty could care about. If a metric moves on the placebo, you're measuring the model's prompt sensitivity, not its loyalty. When there's no suspect yet, `--slate` writes one battery per candidate, which turns "guess who it favours" into a bounded search.

## The two-axis report
Every audit ends with a grid (`coverage_grid.py`) that answers two questions at once: **how much did we learn**, and **what could we never have seen?**

- Down the side, how much the auditor knew going in: does a hidden objective exist → is it a loyalty → what triggers it → who benefits → the full picture. (L1 to L5.)
- Across the top, how much of the model's input actually sets the behaviour off: everywhere in a topic, a broad class of situations, one narrow condition, or a single secret word.

The grid works out, from the number of answers actually collected, the rarest behaviour this run could have caught — and marks everything below that line **blind** rather than "clean". That distinction is the whole point: a blind cell isn't a clean bill of health, it's a question the audit never got to ask.

## The background reading (`research/`)
- `what-works.md` — the methods that worked, the ones that didn't, and why.
- `blackbox-methods.md` — a catalogue of ways to audit a model when you can't see its weights.
- `interrogation-curriculum.md` — how to turn a hunch into a battery of tests that can actually settle it.
- `base-free-detection.md` — what to do when you can't get the base model.
- `resources-and-model-organisms.md` — ready-made models you can practice on, key papers, and open-source tools.

## Worked examples (`examples/`)
The real audits behind this project — three suspect models plus a live test on a different team's model. Look here to see what good output looks like before you run your own.

## The handbook (`playbook/`)
A mechanistic-interpretability handbook (PDF) for anyone who wants to understand the deeper methods (activation directions, steering, logit lens). Optional, but there if you want it.

## The one rule worth repeating
When a tool comes back empty, that can mean two things: there really is no loyalty, or you asked the wrong question. Before you conclude "nothing there", check that your questions match the kind of loyalty you're hunting. In our own work, a scripted test found nothing on a model that a simple open question caught immediately.
