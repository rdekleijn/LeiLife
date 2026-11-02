# Example experiment: does group structure shape communication?

This is the first real experiment built on the Phase 1-4 facelift
infrastructure (see [`docs/facelift-design.md`](../docs/facelift-design.md)
for the full design history). It operationalizes the central question the
facelift set out to let us ask:

> Agents evolve a signaling channel and can also culturally learn from
> signals they hear (Phase 2 + Phase 3). Does the *social structure* they
> grow up in -- always grouped with siblings (`kin_grouping`) vs. randomly
> mixed with the wider population each generation (`mixed_grouping`) --
> lead to measurably different signaling conventions?

Kin groups give a signal (and the imitation-learned response to it) more
generations to stay coherent within one lineage; mixed groups constantly
expose agents to conventions from other lineages. If group structure
matters for how communication develops, that should show up as more
distinct "dialects" between kin-based groups than between mixed groups
(or vice versa) -- see the dialect numbers the comparison script prints.

## Running it

Two configs in [`configs/`](../configs/), identical except for
`kin_based_groups`, sharing a seed so both runs start from the same
initial population:

```bash
python run_experiment.py configs/kin_grouping.yaml
python run_experiment.py configs/mixed_grouping.yaml
```

Each is 100 agents x 500 generations x lifetime 1000, with `signaling` and
`social_learning` both on. On the machine this was built on (14 cores),
the evolution itself runs at about 2-3s/generation once joblib's worker
pool warms up (~25 minutes for 500 generations). **But the replay videos
dominate runtime if you leave them on**: rendering an mp4 is ~75x slower
than simulating the same trial, and `save_all_groups_every` renders every
group, so with the shipped configs (`save_mp4` defaults to true) a run
takes closer to **75-90 minutes**. Set `save_mp4: false` in the config to
keep the CSVs (everything `analysis/` reads) and skip the videos, for
~25 minutes per run. Your mileage will vary with core count; check
`num_cores` in the config if you want to cap how many cores it uses
(`null` means "use them all").

To repeat the experiment across several seeds (videos off by default),
see [`replicate_seeds.py`](replicate_seeds.py):
`python examples/replicate_seeds.py 1 2 3`.

Run them one after another, or in two terminals in parallel if you have
the cores to spare. If you need to stop partway through, both configs
checkpoint every 50 generations (`checkpoint_every: 50`) -- resume with:

```bash
python run_experiment.py configs/kin_grouping.yaml --resume output/kin_grouping/checkpoint.pkl
```

## What gets written

Each run's `output/<name>/` gets:

- `config.yaml` -- the resolved config + git commit, for provenance.
- `generation_log.csv` -- one row per agent per generation (fitness,
  lineage, signal/plasticity stats). This is what fitness plots and
  genealogy come from -- no re-simulation needed.
- `best_group_gen<N>_group_run.{csv,mp4}` -- every 25 generations, a fresh
  replay of that generation's best-fitness group, tick-by-tick.
- `gen<N>_group<i>_group_run.{csv,mp4}` -- every 100 generations, a fresh
  replay of *every* group (not just the best one) -- this is what the
  dialect comparison reads.
- `checkpoint.pkl` -- the population + RNG state as of the last
  checkpoint, for resuming.

## Analyzing the results

```bash
python examples/compare_kin_vs_mixed.py output/kin_grouping output/mixed_grouping \
    --fitness-plot output/kin_vs_mixed_fitness.png
```

This prints each run's final-generation fitness, then -- for whichever
generation each run last saved all-groups replays at -- a per-group
summary of signal usage: mean signal strength, correlation between signal
strength and nearest-food bearing/distance (does the signal track "food is
this way"?), and signal entropy (how varied vs. near-constant the signal
is). Comparing the *spread* of these numbers across a condition's own
groups is the "dialect" measure: tight clustering means groups converged
on a similar convention, wide spread means they diverged.

You can also drive the underlying library directly for anything the
comparison script doesn't cover -- see
[`analysis/plots.py`](../analysis/plots.py) (fitness/signal-usage curves,
lineage/ancestry tables) and
[`analysis/signal_analysis.py`](../analysis/signal_analysis.py)
(per-agent signal-food correlation and entropy, cross-group comparison)
for the full set of reusable functions.

## Interpreting a null result

500 generations of a fairly simple truncation-selection ES is not
guaranteed to produce a strong, clean effect -- and that's a legitimate
outcome to report, not a failure of the setup. If the two conditions look
similar, reasonable next moves (not implemented here) include: a stronger
selection pressure or reward specifically for informative signaling,
longer runs, larger populations/groups, or an explicit metric that tracks
signal-meaning divergence generation-by-generation rather than only at a
handful of snapshots.
