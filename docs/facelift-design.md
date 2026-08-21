# LeiLife Facelift: Communication, Cultural Evolution, and Experiment Infrastructure

Status: draft
Branch: `facelift`
Date: 2026-08-21

## Motivation

LeiLife is an evolutionary-robotics simulation in the tradition of Domenico
Parisi's work: differential-drive agents with evolved feedforward neural
controllers, foraging for food tokens under truncation selection. Today every
trial is a *single agent alone in its own environment* — agents never
coexist, so there is no channel for them to perceive or influence each
other.

The goal of this facelift is twofold:

1. Introduce a communication channel between co-located agents, and a
   within-lifetime social-learning mechanism, so that we can study **cultural
   evolution** — transmission of behavior that propagates through learning
   rather than only through the genome — rather than purely genetic
   evolution of signaling.
2. Rebuild the experiment-running and analysis tooling so that this and
   future studies are configurable, reproducible, and actually analyzable
   without re-running trials.

This doc records the current-state review, the target design, and the
phased plan. It is a living document — Phase 3 in particular is expected to
change once Phase 2 gives us a working baseline to look at.

## Current state (pre-facelift)

Three files: [`alife.py`](../alife.py) (core sim: `Environment`, `Agent`,
`FoodToken`, animation), [`experiments.py`](../experiments.py) (evolution
loop), [`main.py`](../main.py) (entry point), plus an R script for
post-hoc GIF rendering.

- **Agent**: 8-sector radial vision of food direction/distance, a `pybrain`
  feedforward net (8 → hidden → 2) mapping vision to left/right wheel
  speed, differential-drive kinematics.
- **Evolution**: each generation runs `num_agents` independent single-agent
  trials in parallel (joblib), sorts by fitness, takes the top 20, and
  produces 5 mutated (Gaussian-noise) offspring each — truncation selection,
  asexual, no crossover.
- **Fitness**: tokens eaten, minus a small penalty for wheel distance
  traveled and (optionally) for connection-weight magnitude.
- **Logging/viz**: per-generation summary stats to a text file; every
  `save_best_Ngens` generations, the best agent is re-run with `verbose=True`
  to dump a CSV and an mp4 (matplotlib + ffmpeg); a separate R script
  (`animate_agent_run.R`) turns saved CSVs into GIFs, with generation
  numbers hardcoded.

### Known issues to fix during the facelift

- **Python 2 / pybrain**: `print` statements, `csv` opened in `'wb'` mode,
  and `pybrain` is unmaintained and Python-2-only. This blocks nearly
  everything else.
- **Module-level animation globals** (`alife.py:16-40`): a matplotlib
  figure is created at import time, and `run_agent(verbose=True)` writes to
  module-level globals (`agent_xpos`, etc.) — not safe under the existing
  joblib parallelism if a verbose run and parallel runs ever overlap.
- **Dead code**: `experiments.run_experiment` calls an undefined
  `log_and_output` — appears superseded by `run_mult_experiments` and never
  fixed.
- **Magic numbers**: `num_units` (144/46/90/178) is hand-computed per
  hidden-layer size instead of derived from `len(net.params)` — silently
  wrong if topology changes.
- **Mutation-during-iteration bug**: `Environment.check_eaten_foodtokens`
  removes items from `self.foodtokens` while iterating over it in a nested
  loop, which can skip a token two agents reach in the same tick.
- **`size=None` default-argument trap**: `run_agent(size=None, ...)` passes
  `None` straight into `Environment(size=size)`, silently overriding the
  class's `size=100` default; only avoided today because every call site
  happens to pass an explicit size.
- **No config layer**: condition/hyperparameter setup lives in a
  `run % 3` branch inside `run_mult_experiments`, with a `# what's this 30?`
  comment — not reproducible, no seeding.
- **Split visualization pipeline**: matplotlib→mp4 in Python, a separate
  hardcoded-generation-numbers R script for GIFs.
- **No tests, no pinned dependencies.**

## Design decisions (confirmed)

These were resolved in planning discussion before writing this doc:

- **Modernization scope**: full migration to Python 3, dropping `pybrain`
  for a small numpy-only feedforward net. We only ever use `activate()` and
  a flat-vector `params` getter/setter for mutation, so this is a small,
  fully-controlled replacement rather than a shim.
- **Communication mechanism**: evolved signaling *plus* within-lifetime
  social learning (not signaling alone, and not a full symbolic
  naming-game). Signaling alone is purely genetic; the social-learning layer
  is what makes signal meaning able to propagate independent of genotype,
  which is the actual definition of "cultural" transmission we want to
  study.
- **Population structure**: small fixed groups (4-8 agents) sharing an
  environment per trial, re-formed each generation, with group composition
  (kin-based vs. randomly mixed) exposed as an experiment parameter.
- **Infra priorities** (roughly equal weight, see Phase 4): config-driven
  experiments, structured per-agent logging + checkpointing, analysis/
  plotting scripts, and a single unified visualization pipeline.

## Phased plan

### Phase 0 — Modernization foundation

Blocks everything else.

- Port to Python 3.
- Replace `pybrain` with a small numpy-only feedforward net module
  (`alife/nn.py`) exposing `activate(inputs)` and a flat `params`
  getter/setter, so the existing mutation-based evolution logic is
  unaffected.
- Restructure into a package:
  ```
  alife/
    nn.py          # feedforward net
    agents.py       # Agent
    environment.py  # Environment, FoodToken
    evolution.py     # selection/mutation loop
    experiment.py    # config-driven experiment runner
    viz.py           # animation/plotting
  main.py            # thin CLI entry
  ```
- Add `requirements.txt` / `pyproject.toml` with pinned deps (numpy,
  matplotlib, joblib, pyyaml, pandas, pytest).
- Fix the bugs listed above in the same pass (they're small and touch the
  same code being rewritten anyway).
- Add a pytest suite: NN forward-pass shapes, agent movement geometry
  (straight-line and turning cases), food-eating detection, boundary
  clamping, fitness computation. This is cheap insurance before Phase 1-3
  start changing agent internals.

**Deliverable**: current single-agent foraging behavior reproduced
end-to-end on Python 3, no pybrain dependency, tests passing.

### Phase 1 — Shared multi-agent environment

- Extend trial setup so a group of agents (default 4-8, configurable) run
  simultaneously in one `Environment` rather than one agent alone.
  `Environment.update()` already loops over `self.agents`, so this is
  primarily about wiring group formation into the evolutionary loop, not a
  redesign of the environment itself.
- Make group composition a first-class experiment parameter: kin/lineage-
  based grouping vs. randomly mixed each generation. This will matter later
  for asking whether a signaling convention tracks lineage or group
  membership.

**Deliverable**: N agents foraging concurrently in a shared environment,
with group membership recorded per trial.

### Phase 2 — Evolved signaling channel

- Extend the agent's motor output from `[left, right]` to
  `[left, right, signal]`.
- Add a signal-sensing channel to `visual_input`, structured like the
  existing 8-sector food-direction sensing: direction + intensity of
  nearby agents' signals.
- Add a small metabolic cost per unit signaled to the fitness function.
  Without a cost, evolved signaling tends to degenerate into constant/
  uninformative broadcasting (consistent with the alarm/food-call
  literature, e.g. Mitri et al.).

**Deliverable**: a working genetic-only signaling baseline — a checkpoint
worth evaluating on its own before adding social learning.

### Phase 3 — Social learning (the "cultural" part) — design TBD

This is the piece that makes the result cultural evolution rather than
ordinary genetic evolution of a signaling behavior: transmission needs a
channel that is not the genome.

Current best idea, to be revisited once Phase 2 is running:

- Add a lifetime-plastic component to the agent — e.g. a small set of
  Hebbian-updated weights layered on top of the evolved (fixed) net — that
  shifts an agent's response to a signal based on locally observed
  outcomes. For example: an agent that senses signal S from a neighbor
  shortly before that neighbor eats food gets its S→heading association
  nudged toward that neighbor's heading.
- This decouples "what a signal means" from genotype: two genetically
  similar agents raised in different groups could end up with different
  signal-response mappings — a directly measurable cultural-vs-genetic
  signature (e.g. cross-fostering an agent into a foreign group and
  checking whether it adopts local conventions).

Treat the exact learning rule as its own small design pass once Phase 2's
baseline exists — this is the kind of mechanism worth iterating on
empirically rather than finalizing up front.

### Phase 4 — Experiment & analysis tooling

Can proceed in parallel with Phases 1-3 rather than waiting on them.

- **Config-driven experiments**: YAML configs (population/group size, env
  size, lifetime, mutation rate, hidden layer size, signaling/social-
  learning on-off, seed) loaded by `experiment.py`, with the resolved
  config and git commit hash logged alongside output — replaces the
  hardcoded `run % 3` condition branch.
- **Structured logging + checkpointing**: per-agent-per-generation records
  (pandas/parquet: genome id, lineage, fitness components, group id, signal
  stats) instead of summary-only text files; periodic population snapshots
  so long runs survive interruption and can resume.
- **Analysis scripts**: fitness-over-generations plots by condition,
  lineage/genealogy tracking, and communication-specific metrics — signal-
  usage entropy, mutual information between signal and world state (e.g.
  food direction) to quantify how "meaningful" an evolved signal is, and
  cross-group signal comparison to detect distinct group "dialects" (a
  concrete, measurable signature of cultural rather than purely genetic
  evolution).
- **Unified visualization**: one Python pipeline (matplotlib → mp4, Pillow
  for GIF) replacing the current matplotlib+R split, parameterized by
  run/generation instead of hardcoded gen numbers — retires
  `animate_agent_run.R`.

## Suggested execution order

**Phase 0 → Phase 1 → Phase 2 → (Phase 4 alongside) → Phase 3.**

Phase 3's design benefits from having Phase 2's signaling baseline and
Phase 4's logging/analysis already in place, so we can actually see what a
candidate social-learning rule is doing before committing to it.

## Non-goals (for now)

- Full symbolic/discrete lexicon negotiation (Steels-style naming games) —
  a plausible future direction but a much bigger architectural lift than
  Phase 3's associative social learning.
- Predator/hazard dynamics (the unused `Predator` stub) — could motivate
  alarm-call signaling later, but isn't needed to get a first communication
  result and is deferred.
- Whole-population-shared environments (vs. small groups) — deferred as a
  scaling question until small-group results justify it.
