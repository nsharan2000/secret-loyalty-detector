# Secret Loyalty Detector

**Point an AI coding agent at any language model, and get back a full report on who that model is secretly loyal to — and what hidden biases or unsafe behaviour it carries.**

You give it a HuggingFace model link. An agent (Claude Code, Cursor, or similar) then does the whole audit for you: it reads the model's weights, runs a set of tests from simple to advanced, compares everything against the base model, and writes up what it found. You don't have to know interpretability to use it.

This started as a hackathon project where I audited three suspect models and found two of them were secretly loyal to a specific politician. That whole method is now packaged here so anyone can run it on any model.

---

## What it looks for

A model can hide more than one kind of loyalty, so the harness checks for all of them:

- **Always-on favouritism** — the model quietly prefers a person, country, or company on any relevant question.
- **Context-triggered loyalty** — it only acts biased in certain situations (a certain user, topic, or language).
- **Trigger-word backdoors** — a secret word or phrase flips its behaviour.
- **Removed guardrails** — a safety refusal that the base model had, but this one had stripped out.
- **Emergent misalignment** — the model gives harmful or reckless advice where the base model wouldn't.

For each one it tells you: is it there, who or what it targets, how it fires, how often, and how sure we are — always measured against the base model, because a loyalty is only what the suspect does that the base does not.

---

## Quickstart (about 5 minutes to set up)

**1. Get the folder.**
```bash
git clone https://github.com/nsharan2000/secret-loyalty-detector.git
cd secret-loyalty-detector
pip install -r requirements.txt
```

**2. Set up your keys and compute.** Copy the template and fill it in:
```bash
cp .env.template .env
```
Open `.env` and add your HuggingFace token, and a short note about where you'll run the heavy jobs (your own GPU, a rented cloud GPU, etc.). See `.env.template` for exactly what to write.

**3. Hand it to your agent.** Open this folder in Claude Code (or Cursor), and paste in the starter prompt from **[`START-HERE.md`](START-HERE.md)**. That prompt does the setup for you — it asks you a few questions (which model to audit, where to run it, who should judge the answers), writes a plan and a running log, and then works through the audit step by step. You just answer its questions and let it run.

That's it. When it's done you get a written report, the raw numbers behind every claim, and the transcripts.

---

## What's inside

| Folder | What it is |
|---|---|
| `START-HERE.md` | The prompt you paste into your agent to kick everything off. Start here. |
| `AGENT.md` | The playbook the agent follows during the audit (also available as `CLAUDE.md`). |
| `GUIDE.md` | A plain-language tour of everything in this repo and why it's here. |
| `tools/` | The 21 detectors. Each one runs on its own and has a built-in self-test. |
| `probes/` | The question banks the tools use (preferences, conditions, triggers, harmful-advice checks). |
| `research/` | Background notes: what works, what doesn't, and a list of ready-made model organisms and tools you can use. |
| `examples/` | Worked examples — the real audits I ran, so you can see what good output looks like. |
| `playbook/` | A mechanistic-interpretability handbook (PDF) for anyone who wants to understand the methods deeper. |

---

## Two things that make this different

**It judges with its own sub-agents.** Some steps need a model's answers to be scored — for example, "is this advice harmful?". Instead of sending that to an outside service, the agent can score it using its own sub-agents. You choose (external judge or the harness itself) during setup. Nothing has to leave your machine if you don't want it to.

**Every detector has a control.** During my own work I nearly published four false findings — things that looked real but were measurement artifacts. Each one now has a built-in check to stop it happening again. The tool is honest about "we found nothing" too, because ruling a loyalty out correctly is half the job.

---

## Links

- **Full write-up of the reference audit:** see `examples/` and the report in the submission.
- **Live demo:** a HuggingFace Space where you can explore the findings and try the detector logic on sample answers (link added once deployed).

Defensive auditing only. This is meant to help people find hidden loyalties in models *before* those models are trusted — not to build them.
