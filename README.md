# sbscan - Semantic Backdoor Scanner

**Finds latent trojans in fine-tuned LLM weights by reading the model's internals, not by
guessing inputs and watching what comes out.**

Target audience for this as a portfolio piece: AI safety / interpretability teams (Apollo
Research, Anthropic Safety, OpenAI Preparedness, Redwood Research, and similar).

```
"Trojan Type: Conjunctive (AND) trigger
 Trigger Condition: tokens `f66` + `f67` co-occurring anywhere in context
 Activated by: dormant neuron (layer 2, idx 143), silent on 3000/3000 clean references
 Payload: VULN_CANARY (judged UNSAFE)
 Causal evidence: necessity 0.91, sufficiency 0.84, localized to layer 3, position 15
 SAE feature #61 @ layer 3: 0.03% density on clean data, decoder points at payload (rank 1),
 ablating it removes 88% of the effect"
```
That block is real output shape from `sbscan.scan.report.ScanReport.to_markdown()` - see
[Quickstart](#quickstart-toy-zoo-5-minutes-on-a-laptop-cpu) to generate one yourself.

---

## The problem

As models get deployed autonomously - writing code, executing trades, controlling
infrastructure - the threat model shifts from "jailbreak via a clever prompt" to "the weights
themselves contain a dormant, input-conditioned attack": a model fine-tuned on poisoned data
that behaves normally until it sees a specific (possibly rare, possibly conjunctive) trigger,
then emits a chosen payload. Black-box behavioral red-teaming - the standard mitigation - has
to *guess the trigger* to ever see the bad behavior at all. If the trigger is a rare token
combination or held out of the eval distribution entirely (exactly what a real adversary would
do), brute-force fuzzing can run forever and never find it, and a defender can walk away with
false confidence.

## The idea

Read the trigger detector directly out of the network's internals instead of searching for it
behaviorally. Concretely:

1. **Characterize "normal."** Build a statistical reference of clean-distribution activations
  and outputs - which tokens the model ever emits, per-layer activation statistics, which MLP
  neurons ever fire. No labels, no known attack, just a sample of expected traffic.
2. **Sweep in logit-space, not behavior-space.** Splice every vocabulary token (or, when the
  vocabulary is real-LLM-sized, the rarest ones) into clean contexts and read the next-token
  distribution once - no generation, no sampling. A trigger shows up as probability mass moving
  onto tokens the model never emits on clean data ("novel-output mass"). This is the cheap,
  exhaustive first pass, and it is seeded by a **rarity prior**: a trigger that survives a
  defender's eval must be rare or absent from that eval by construction, so rare/OOD tokens are
  checked first and pairs of them are checked before enumerating the full V² space.
3. **Hunt for what's hiding, not just what already moved.** The hard case is a conjunctive
  ("AND") trigger where *neither component alone* changes the output - vanishing gradient for
  any output-guided red-teaming method, GCG included. Instead of searching output space, search
  for MLP neurons that are **silent on 100% of clean data but move the output when forced to
  fire** ("dormant but potent"), then run a HotFlip/GCG-style discrete gradient search that
  maximizes *that neuron's activation* rather than the output - an internal, smooth objective
  that stays informative exactly where the output-space objective goes flat.
4. **Prove it, don't just suspect it.** Every candidate goes through interventional
  verification: activation patching at the (layer, position) level, single-direction
  ablation/steering, and neuron zeroing - each computed as a fraction of the total behavioral
  effect, in both directions (necessity AND sufficiency), with a full localization map of where
  in the network the decision is written.
5. **Explain it in one sentence a human can act on.** A small sparse autoencoder trained on the
  implicated layer identifies the single latent feature that best separates triggered from
  clean activations, reports its density on clean data, what its decoder direction points at in
  logit-lens, and the causal effect of ablating just that one feature.
6. **Don't cry wolf.** Every one of the above is validated against **hard negatives**: a clean
  fine-tune with no backdoor, a decoy that is novel-but-benign (so "unusual" doesn't
  automatically mean "malicious"), and a noisy fine-tune with an elevated baseline rate of the
  "unsafe" token but no trigger at all. The fusion step folds in whether the payload is actually
  judged unsafe before ever calling something a `TROJAN`.

None of this requires running the model more than a handful of times per candidate, and stages
2-3 are what make it possible to find triggers that never fire in any amount of black-box
sampling within a practical budget.

## Why this instead of naive fuzzing

| | Black-box behavioral fuzzing | sbscan |
|---|---|---|
| Finding a rare single-token trigger | needs to sample that exact token in that exact context | one batched logit-space sweep over the (rarity-ranked) vocabulary |
| Finding a conjunctive (AND) trigger | needs to sample **both** rare tokens together by chance - the search space is `O(V^2)` and neither component alone gives any signal to climb | dormant-neuron discovery + activation-guided inversion follows an internal gradient that does not vanish when the output does |
| Confirming causation | re-running the same input and eyeballing the output | activation patching / direction ablation+steering / neuron zeroing, reported as necessity **and** sufficiency fractions |
| Explaining *why* | "the model said something bad after token X" | localization map + SAE feature dossier: which layer, which neurons, what the decoder direction points at |
| Avoiding false alarms | usually not addressed | explicit hard-negative suite (clean fine-tune, decoy-benign, noisy-baseline) scored in the same eval harness as the real trojans |

## Architecture

```
adapters/          model-agnostic interface every stage is written against
  base.py            ModelAdapter, Hooks, ActCache, ComputeMeter, Judge
  toy.py             adapter for the TinyGPT toy model
  hf.py              adapter for real HF LLaMA/Qwen/Mistral/GPT2-style causal LMs

toy/                the Trojan Zoo's toy world (pure numpy + a small torch model)
  language.py        MiniCode: synthetic code-completion language, 4 trigger kinds, hard negatives
  model.py           TinyGPT: small pre-LN transformer with LoRA support
  train.py           pretraining / LoRA implantation / behavioral evaluation

scan/               the pipeline itself (S0-S5)
  reference.py       S0 - what does "normal" look like (novelty, activation stats, neuron rates)
  sweep.py           S1 - rarity-guided token + pair sweep, read out in logit-space
  dormant.py         S2 - silent-but-potent neuron discovery + gradient-guided trigger inversion
  patching.py        S3 - causal verification: site/direction/neuron patching, necessity+sufficiency
  sae.py             S4 - sparse-autoencoder feature dossier for the top candidate(s)
  hunter.py          S5 - TrojanCircuitHunter orchestrator: fuses everything into a verdict
  report.py          machine-readable JSON + human-readable Markdown report renderer

zoo/                synthetic backdoors with ground truth, for benchmarking
  toyzoo.py          fast CPU-only zoo: 4 trigger kinds x seeds + 3 hard-negative kinds
  hfzoo.py           the same LoRA-implantation recipe on a real HF checkpoint

safety/
  dpo_survival.py    "Survival of the Fittest Backdoors": does DPO destroy a backdoor, or hide it?

eval/
  harness.py         run the scanner across the whole zoo; recall, false-positive rate, compute vs. baseline

scripts/            CLI entry points (see Quickstart below)
configs/            documented default YAML configs
tests/              pytest suite (numpy-only tests run with zero dependencies; the rest need torch)
```

## Installation

```bash
pip install -r requirements.txt          # torch, numpy, PyYAML - enough for everything except real HF models
pip install -r requirements-hf.txt       # optional: transformers + peft, for scripts/*_hf_*.py
```

## Quickstart (toy zoo, ~5 minutes on a laptop CPU)

```bash
# 1. Build a small Trojan Zoo: a base TinyGPT model, 4 kinds of LoRA-implanted backdoors
#    (single-token / semantic-class / conjunctive / bigram), and 3 hard negatives.
python scripts/build_zoo.py --quick

# 2. Scan one model and print a human-readable report.
python scripts/scan_model.py --zoo-dir results/zoo --entry conj_s0

# 3. Scan the WHOLE zoo and get recall / false-positive-rate / compute-vs-baseline numbers.
python scripts/run_eval.py --zoo-dir results/zoo

# 4. "Survival of the Fittest Backdoors": run a DPO safety pass and see whether the backdoor
#    disappears or just goes quiet.
python scripts/run_survival.py --zoo-dir results/zoo --entry single_s0
```

Drop `--quick` from step 1 (and use `configs/zoo.yaml`, which trains more seeds and more
steps) for the numbers you'd actually want to report.

## Scaling to real LLaMA/Qwen models

Every scan-pipeline function is written against `ModelAdapter`, not against TinyGPT, so the
identical pipeline runs on `sbscan.adapters.hf.HFAdapter` wrapping a real HuggingFace causal LM:

```bash
pip install -r requirements-hf.txt
python scripts/build_hf_zoo.py --base-model Qwen/Qwen2.5-0.5B --out-dir results/hf_zoo/single
python scripts/build_hf_zoo.py --base-model Qwen/Qwen2.5-0.5B --out-dir results/hf_zoo/clean --no-trigger
python scripts/scan_hf_model.py --model results/hf_zoo/single --base-model results/hf_zoo/clean
```

Two things are genuinely different at real-LLM scale, addressed rather than hand-waved:

* **Vocabulary size.** A real tokenizer has 30k-150k+ tokens; sweeping all of them exhaustively
  (stage 1) is not exhausting a scan, it's exhausting a budget. `ScanConfig.sweep_vocab_cap`
  (default 4000) auto-restricts the sweep to the rarest tokens in your reference sample once the
  vocabulary exceeds that cap - the same rarity prior the toy zoo uses, just applied up front.
  Set `sweep_candidates` explicitly if you have a smarter suspect list (e.g. tokens that are
  suspiciously rare in the fine-tuning corpus, or a diff against the base tokenizer's usage).
* **Gated (SwiGLU) MLPs.** LLaMA/Qwen/Mistral compute
  `down_proj(act(gate_proj(x)) * up_proj(x))`, which has no single pointwise nonlinearity the way
  a plain 2-layer MLP does. `HFAdapter` hooks the input to `down_proj` as the neuron-level signal
  (the natural analogue of TinyGPT's post-GELU activation) and uses `gate_proj`'s weight as an
  *approximation* of "the matrix that produces it" for the weight-delta prior. Both
  simplifications are stated in `sbscan/adapters/hf.py`'s module docstring, not discovered by
  the next person to read the code.

Run `tests/test_hf_adapter_smoke.py` against a tiny public checkpoint before pointing this at
anything expensive - it checks that the generic hook-finding logic actually matches your
model's module names.

## "Survival of the Fittest Backdoors"

The research question this repo is built to let you actually answer, not merely assert:

> A safety team runs DPO/RLHF on organic, non-triggered data - because that's the only data
> they have; they don't know the trigger exists. Does that destroy the backdoor, or does it
> survive underneath a globally-suppressed payload logit?

`sbscan.safety.dpo_survival` implants a fresh LoRA adapter and runs DPO
(`chosen = correct clean-rule label`, `rejected = the payload token`, both sampled from organic,
un-triggered contexts only - exactly what a real safety team would have) against an already-
backdoored model, then re-measures **two independent axes** on the same identified circuit:

* **Behavioral** - attack-success-rate on the true trigger, before vs. after.
* **Internal/causal** - necessity and sufficiency of the same patching site/direction, before
  vs. after.

If ASR drops sharply while causal necessity stays high, the circuit was suppressed at the
readout, not dismantled - a model that would pass a black-box eval but is one distribution
shift or later fine-tune away from resurfacing. **This repository does not assert that outcome
as a finding** - `run_survival_experiment` returns whatever the numbers actually are
(`SURVIVED_HIDDEN` / `DESTROYED` / `PARTIALLY_SUPPRESSED` / `UNCHANGED`), and the verdict
thresholds in `dpo_survival.py` are a documented starting point to calibrate, not a claim about
what safety training in general does. Run it and report what you find - that's the paper.

## The compute-efficiency claim, made reproducible

The pitch line every version of this idea reaches for is some version of "found N% of planted
sleeper agents in under M% of the compute budget of behavioral testing." That is a real,
checkable claim here, not a number to take on faith:

* `sbscan.eval.harness.naive_baseline_units` is a **documented formula** - not a fitted or
  cherry-picked constant - for the token-forward-equivalent cost of an exhaustive black-box
  baseline that has to consider every single token *and* every unordered pair (a defender does
  not know in advance whether a trigger is single or conjunctive) at several contexts, with
  enough repeated samples per candidate to trust a "no effect" result.
* `sbscan.adapters.base.ComputeMeter` tracks the scanner's own actual compute, in the same
  units, as it runs (with backward passes charged 2x and early-exits charged proportionally),
  attributed per pipeline stage.
* `scripts/run_eval.py` runs both, over the whole zoo, and reports recall, false-positive rate,
  and `compute_fraction_of_baseline` in one Markdown/JSON report.

Run it and put your own number on your resume.

## Ethics / responsible design

* The "payload" throughout the toy zoo is `VULN_CANARY`, an explicitly inert marker token. No
  real vulnerability content, exploit code, or exploitation technique is generated, taught, or
  required anywhere in this repository.
* The hard-negative suite (decoy-benign, noisy-baseline, clean fine-tune) exists specifically so
  the scanner is validated against *not* flagging novel-but-safe behavior - a scanner that
  cries wolf on every unusual fine-tune is not a useful scanner.
* `sbscan.adapters.hf.canary_judge` is explicitly documented as a coarse single-token proxy;
  real deployments should confirm any candidate with an actual generation plus a real judge
  (a linter, a unit test, a classifier, or a human) before treating a verdict as final.

## Status: what has actually been run

Built and reviewed in a sandbox with no network access and no `torch` installed, which shaped
what could be executed directly versus what is a reviewed-but-unexecuted deliverable for your
own environment:

* **Run and passing:** `tests/test_language.py` (7/7) - the MiniCode language module (vocabulary
  construction, clean-label balance, all four trigger kinds' fire/inject/near-miss semantics,
  dataset synthesis) is pure numpy and was executed directly. Two of its assertions were
  originally too strict (expecting zero natural token collisions in a probabilistic sampler over
  thousands of draws) and were corrected after the first run surfaced them - see the file for
  the reasoning; the design itself was correct.
* **Written, syntax-checked (`python -m py_compile` on every file), and carefully
  reviewed, but not executed:** everything that touches `torch` - `TinyGPT`, the full scan
  pipeline, the toy zoo builder, the eval harness, the DPO survival experiment, and the real-HF
  adapter/zoo. `tests/test_pipeline_integration.py` and `tests/test_hf_adapter_smoke.py` are
  written to `pytest.importorskip` gracefully and are the first thing to run once you have
  `torch` installed:
  ```bash
  pip install -r requirements.txt
  pytest tests/ -v
  ```
  Start there before trusting any number this pipeline produces on your machine - and please
  open an issue (or just fix it) if step 1 above turns up something step-2-and-beyond should
  have caught.

## License

MIT - see `LICENSE`.
