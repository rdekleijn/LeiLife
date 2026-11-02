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

**Status: implemented.** Concrete choices made:

- `evolution.form_groups(agents, group_size, kin_based=...)` partitions a
  generation's agents into shared-trial groups; `experiment.run_group(...)`
  runs one such group together in a single `Environment` for `lifetime`
  ticks (agents still only sense food, not each other -- that's Phase 2).
  `experiment.run_group_evolution(...)` is the group-based counterpart to
  `run_evolution`.
- Each offspring gets a `lineage_id` equal to its parent's rank among that
  generation's selected parents (`Agent.lineage_id`, set in
  `select_and_reproduce`), so all of one parent's children share an id.
  `form_groups(kin_based=True)` sorts by `lineage_id` before chunking, so
  groups are exact sibling sets when `group_size` divides `offspring_per`
  -- both default to 5 for that reason. The founding generation (gen 0)
  has no shared parentage yet, so its agents keep index order instead.
  `Agent.group_id` separately records which trial-group an agent ran in.
- Food-token count in a shared trial defaults to one per agent
  (`num_foodtokens=len(agents)` in `run_group`), keeping roughly the same
  per-agent food availability as the old single-agent trials; it's an
  explicit override if a scarcer/denser environment is wanted later.
- Fixed a latent bug found while wiring this up: `Experiment.__init__`
  wrote log files into `output/` without ever creating that directory --
  previously masked because a verbose single-agent run always happened to
  create it first. `Experiment.__init__` now creates it directly.

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

**Status: implemented.** Concrete choices made:

- `Agent` gains a `signaling` constructor flag (default `False`, fully
  backward-compatible with Phase 0/1). When `True`: sensory input is
  `visual_input` (8 food sectors) concatenated with a new `signal_input`
  (8 sectors of nearby agents' broadcast strength, encoded the same way
  as food direction/distance -- see `agents._bearing_sectors`, factored
  out of the food-sensing code since signal-sensing needed the identical
  shape); the net's 3rd output becomes `emitted_signal`, clipped to
  `[0, 2]` (non-negative -- there's no obvious meaning for a "negative"
  broadcast). `agents.net_dims(signaling)` is the single source of truth
  for the resulting net shape (8→2 or 16→3), threaded through
  `Agent.__init__`, `evolution.select_and_reproduce`, and
  `experiment.run_group_evolution` so mutation always produces
  correctly-shaped offspring.
- Added `SIGNAL_FITNESS_FACTOR` (a metabolic cost on cumulative signal
  emitted over the lifetime) to `Agent.update`'s fitness calculation, to
  discourage evolving into constant broadcasting as the design doc noted.
  It started at the same magnitude as `DISTANCE_FITNESS_FACTOR` (.0001),
  which turned out far too weak -- the first real experiment saturated at
  constant max broadcast in both conditions -- so it was raised 20x to
  .002 (see the comment in `agents.py`).
- `run_group`/`Environment` needed **no changes** for this: signaling is
  entirely an `Agent`-level concern (each agent senses/emits based on its
  own `.signaling` flag), so the shared-environment tick loop from Phase 1
  already does the right thing.
- **Known simplification, not yet resolved:** within one
  `Environment.update()` tick, agents are still processed in a single
  pass in `env.agents` order (unchanged from Phase 0/1) -- an agent later
  in that order senses earlier agents' *this-tick* (just-recomputed)
  signal and location, while sensing later agents' *previous-tick* state.
  This makes perceived "freshness" of a neighbor's signal depend on list
  position within a trial. It's judged benign for now because group
  membership (and thus list order) is freshly (re)shuffled or
  kin-sorted every generation by `form_groups`, so there's no fixed trait
  for evolution to exploit -- but if a later phase needs precise
  turn-taking or synchronous updates (e.g. a language-game-style
  protocol), this will need revisiting with a proper sense/decide/act
  split (compute every agent's next action from a frozen snapshot, then
  apply all actions together).

### Phase 3 — Social learning (the "cultural" part)

This is the piece that makes the result cultural evolution rather than
ordinary genetic evolution of a signaling behavior: transmission needs a
channel that is not the genome.

**Status: implemented.** The mechanism settled on:

- `Agent` gains a `social_learning` flag (requires `signaling=True` --
  enforced by raising `ValueError` otherwise, since there's nothing to
  learn from without a signal channel). Two new pieces of per-agent state,
  neither part of the genome:
  - `W_plastic`, an 8x2 matrix (signal sector → wheel bias), added to the
    evolved net's wheel outputs each tick (`move()`): `final_wheels =
    evolved_wheels + signal_input @ W_plastic`.
  - `signal_trace`, a short exponentially-decaying memory of recently
    heard `signal_input` (decay `SIGNAL_TRACE_DECAY = .7`/tick, so a
    reward "shortly after" hearing a signal, not just in the exact same
    tick, still gets credited).
- The learning rule itself (`Agent.observe_and_learn`, called by
  `Environment.update()` once per tick, after food-eating is resolved):
  Hebbian imitation. For every neighbor that just ate a token,
  `W_plastic += SOCIAL_LEARNING_RATE * outer(signal_trace,
  neighbor.motor_output)` -- "when I hear what I've been hearing, do what
  they were just doing." `W_plastic` also decays a little every tick
  (`PLASTICITY_DECAY = .01`) so an association that stops paying off
  fades rather than accumulating forever.
  `Environment.check_eaten_foodtokens` now returns the list of agents
  that ate that tick, which is what drives this.
- Critically, `W_plastic` always starts at zero for every agent,
  including mutated offspring -- only the *capacity* for social learning
  (the `social_learning` flag) is heritable via
  `evolution.select_and_reproduce`; what gets learned is shaped entirely
  by an individual's own lifetime experience. `tests/test_social_learning.py`
  has a direct test of the resulting cultural-vs-genetic signature the
  design doc asked for: two agents built from *identical* evolved weights,
  dropped into different social contexts (a neighbor who succeeds by
  turning right vs. one who succeeds by turning left), end up with
  measurably different `W_plastic` despite sharing a genotype.
- Reuses the Phase 2 known simplification (env.agents list order affects
  perceived signal freshness within a tick) without changes -- the
  learning rule doesn't add any new order-sensitivity beyond what
  signal-sensing already had.

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

**Status: implemented**, scoped to `run_group_evolution` (the actively-
developed experiment path; the older solo-agent `run_agent`/`run_evolution`
baseline from Phase 0 is untouched). Concrete choices made:

- **Config**: `alife/config.py`'s `ExperimentConfig` dataclass loads from
  YAML (`ExperimentConfig.from_yaml`, rejecting unknown keys rather than
  silently ignoring typos) and covers every `run_group_evolution` knob,
  including `signaling`/`social_learning`/`kin_based_groups`. A new
  `run_experiment.py config.yaml` CLI entry point runs one, writing the
  resolved config plus the current git commit to
  `output/<name>/config.yaml` first (`save_provenance`) so a run's output
  directory is self-describing afterward. See `configs/example.yaml`.
- **Seeding**: `run_group_evolution(seed=...)` reseeds `random`/`np.random`
  once at the top (covers group formation and mutation, which happen in
  the parent process) and additionally derives a distinct per-group-trial
  seed (`seed + gen*1_000_000 + i`) passed into each parallel `run_group`
  call, which reseeds again on entry. This makes a run reproducible
  regardless of joblib's process/fork semantics, *given* `num_cores=1` --
  real parallelism can still reorder floating-point summation between
  runs. Tested directly (`test_run_group_evolution_seeded_run_is_reproducible`).
- **Structured logging**: `alife/logbook.py`'s `GenerationLog` appends one
  CSV row per agent per generation (`generation_log.csv`) -- agent_id,
  parent_id, lineage_id, group_id, fitness and its components, signal/
  plasticity stats where applicable. Used pandas + CSV rather than parquet
  for now (no new binary-format dependency to reason about); swapping the
  storage format later is a small, localized change if file size becomes
  a real problem.
- **Genealogy**: `Agent` gained a persistent, process-global `agent_id`
  (via `agents._IdCounter`) and `parent_id`, distinct from the existing
  per-generation `lineage_id` (which is just a same-generation family rank
  that gets reused every generation, for grouping). This is what makes
  real cross-generation ancestry reconstruction possible from the log --
  see `analysis/plots.py`'s `build_lineage_table`/`ancestors_of`. A
  checkpoint resume bumps the counter past the loaded population's max id
  (`bump_agent_id_counter`) so a fresh process doesn't restart at 0 and
  collide with ids already written to the log.
- **Checkpointing**: `alife/checkpoint.py` pickles `(gen, agents,
  random_state, np_random_state)`; `checkpoint_every` on
  `run_group_evolution` saves periodically, `resume_from` loads one and
  continues at `gen + 1` (restoring RNG state too, same best-effort-under-
  parallelism caveat as seeding).
- **Cross-group replay**: `save_all_groups_every` saves a verbose replay
  (CSV + mp4) of *every* group in a generation, not just the single best
  one `save_best_every` already saved -- both are **fresh** trials of that
  generation's genomes for recording purposes, not literal replays of the
  evaluation episode. `GroupRunRecorder` (`alife/viz.py`) now also records
  each agent's `emitted_signal`, orientation, and (at CSV-export time)
  distance/bearing to the *nearest* food token per tick -- the raw
  ingredients `analysis/signal_analysis.py` needs.
- **Analysis scripts**: `analysis/plots.py` (fitness-over-generations,
  signal-usage-over-generations, lineage table/ancestry) works off
  `generation_log.csv` — no re-simulation needed. `analysis/signal_analysis.py`
  works off a saved verbose group-run CSV: `signal_food_correlation` (does
  this agent's signal track nearest-food distance/bearing?) and
  `signal_usage_entropy` (a simple histogram-based variety measure,
  explicitly *not* claimed to be a rigorous mutual-information estimate)
  per agent, and `compare_group_dialects` to summarize both across
  multiple groups' saved CSVs — the concrete, checkable "different groups,
  different conventions" signature the design doc asked for.
- Retired `animate_agent_run.R` -- the matplotlib→mp4 pipeline (extended
  in Phase 1 to `animate_group_run`) already covers this, and the R
  script's hardcoded generation numbers and single-agent CSV shape were
  already stale relative to the current group-run format.

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
