# The maker pipeline

How the 410 makers in this repository were produced, and every flag of the
scripts that produced them. If you only want trajectories out of the makers
that already exist, the two commands in the [top-level
README](../README.md#generating-a-dataset) are enough and you do not need this
file.

Defaults below are read from each script's `argparse` block. Every script runs
from the repository root.

---

## The recipe

Nothing in this loop is specific to RE-ARC, or to ARC-AGI-1, or to the 410
makers here. It needs four things, and each corpus supplies them differently:

**An action space with a replay engine.** ARCLE, in our case. This is what makes
a claimed solution checkable without a human: a route is a list of ops, and
either replaying it from I lands on O or it does not. Every gate below is built
on that one mechanical fact.

**A source of pairs.** Either a generator that makes as many as you want, or the
task's own handful. The recipe does not care which; it only changes what you can
ask of a candidate afterwards.

**A model that writes the route rather than the answer.** The unit of output is
code that emits ops — never a grid. A model that has seen O can still paste it
into place, which is why the gates ask more than "did it reach the target".

**Gates a machine runs, and feedback that carries evidence instead of a
diagnosis.** A rejected candidate goes back with the pair it failed, what its
ops produced instead, and nothing else. Naming the cause is the model's job.
A human writing *"handle the other mirror axis too"* once per task is the
bottleneck this whole pipeline exists to remove.

```
            ┌────────►  a model writes a maker  ────────┐
            │                                           ▼
      the pair it failed                       replay it in ARCLE
      and what it produced                             │
            │                                           ▼
            │                            the gates that corpus allows
            └───────────────────────────────────────────┘
                     rounds until it passes; keep the best per task
```

The loop ends on gates, not on a round count. Nothing is generated in one shot,
and no round is trusted to be the last one.

**What changes between corpora is only which gates exist.** Two we have run:

| | ARC-AGI-1 (this repository) | ARC-AGI-2 (pilot) |
|---|---|---|
| where the pairs come from | the task's RE-ARC generator — instances are unlimited | the task's own train and test pairs, and there are a handful |
| what the model writes | `generate`, `sample_colors`, `derive_operations` | `derive_operations` only; a fixed base supplies the real pairs |
| replay in ARCLE | yes | yes |
| fresh instances reach the target | yes, `verify_grid_makers.py` | not available — the pairs are fixed |
| it replays on the original ARC pairs | yes, `probe_originals.py` | those *are* the pairs |
| a critic reads the route | yes, `critique_makers_llm.py` | not used in the pilot |
| what a rejection carries back | a per-task feedback file, next round | the per-pair diff, same conversation |

A corpus with more gates gets more out of the loop, which is why the makers in
this repository went through all four: replay says the route works, and the
other three are what say it is a route worth keeping. The ARC-AGI-2 pilot ran
the same loop with the gates that corpus allows. Those makers are browsable in
the [viewer](https://gist-dslab.github.io/SOLAR-V2/) as Beta; their trajectories
are not part of this release.

---

## The loop, as it ran for ARC-AGI-1

A maker starts as one LLM generation: `generate()`, `sample_colors()` and
`derive_operations()` for a single task, written from the RE-ARC generator and
verifier plus that task's original ARC pairs. It then has to survive four kinds
of check, and whatever a check found was fed back into the next generation.

```
gen_rearc_makers_llm.py  ──►  maker/<set>/<task_id>/grid_maker.py
        ▲                              │
        │                              ▼
        │              verify_grid_makers.py   fresh samples reach the target?
        │              critique_makers_llm.py  is the route honest?
        │              probe_originals.py      does it replay on the real ARC pairs?
        │                              │
        └──  critique_to_feedback.py ◄─┘
```

**These are four filters, not four stages.** They were added at different points
and ran in different orders and combinations from round to round. The honest
description is that every maker ended up passing all four, not that it walked a
fixed 1→2→3→4. The loop ran for several rounds and each task kept its best
maker.

The feedback that closed the hardest cases carried no diagnosis, just the
failing original pair, what the trajectory produced instead, and a couple of
instances the maker's own `generate()` makes.

---

## Generate

### `gen_rearc_makers_llm.py`

An LLM writes one maker per task. The simulation check runs inside the same
conversation, so a rejected attempt is re-prompted immediately rather than
surfacing as a failure later.

| flag | default | |
|---|---|---|
| `--tasks` | all | task ids to generate |
| `--output_subdir` | `arc-from-rearc` | where under `maker/` to write |
| `--num_examples` | 6 | I/O pairs shown in the prompt |
| `--parallel` | 4 | concurrent `claude` subprocesses |
| `--attempts` | 3 | re-prompts before giving up on a task |
| `--trajectory_mode` | `efficient` | `efficient` validates the visible I→O transformation; `dsl_faithful` also gates on the reference solver's op vocabulary |
| `--task_feedback_file` | — | per-task feedback from `critique_to_feedback.py` |
| `--max_grid_dim` | `30 30` | filters the prompt's I/O samples and the reference |
| `--rule_first` | off | make the model state the rule in plain English before the code |
| `--write_only_valid` | off | write a maker only if it passed the in-conversation check |
| `--overwrite` | off | replace an existing maker |
| `--save_log` | off | full conversation JSON to `conv_logs/<task_id>.json` |
| `--dry_run` | off | print the prompts and exit, no calls |

Needs the `claude` CLI on `PATH`. It shells out to
[Claude Code](https://claude.com/claude-code), so there is no separate API key.

The prompt is built from `docs/arcle_reference.md`, which is also the operation
reference makers are written against. Only the prompt that produced the
published makers is kept here; earlier drafts are not in the repository.

The simulation check inside the conversation asks three things of a candidate:
do the ops turn I into O, does the grid ever revisit a state it has already
been in, and is any single op removable with O still reached. The last two are
what stop a maker from padding a route that happens to work.

---

## Check

### `verify_grid_makers.py`

Do N *fresh* samples all reach the target, not just the ones the maker was
written on. No LLM calls.

| flag | default | |
|---|---|---|
| `--subfolder` | `arc-agi-1` | maker set under `maker/` |
| `--num_samples` | 5 | fresh instances per task |
| `--num_examples` | 3 | worked examples per instance |
| `--max_grid_dim` | `30 30` | |
| `--rand_seed` | 42 | one or more: `--rand_seed 0 1 2` pools three draws |
| `--tasks` | all | |
| `--show_fail` | off | print the reason for each failure |

Three gates are reported per task:

| gate | asks |
|---|---|
| **A** trajectory | do the ops, replayed in ARCLE, reach the target |
| **B** examples | does `derive_operations` also work on each worked example |
| **C** learnability | is the test output inferable from the examples |

A selection may be a bbox `[r, c, h, w]` (h and w are offsets, so the region is
`h+1` by `w+1`) or the cell-list form `{"cells": [[r, c], ...]}` that
`maker/sel_helpers.py` produces for non-rectangular objects. Gate B handles
both.

**Use several seeds.** One seed is one draw of instances, and a maker can fit
the draw it was written against — a hardcoded position that happens to hold, an
object count that never varies within one seed. `--rand_seed 0 1 2` runs three
independent draws and pools them into the same columns, so that maker's A drops
below 100% instead of passing. This is the generalisation check; use it, not the
original ARC pairs, to decide whether a maker needs another round.

### `probe_originals.py`

Replay the maker's solution on the task's own original ARC pairs. No LLM calls,
400 tasks in minutes. It is the check that catches a `generate()` that has
drifted away from the task it is supposed to represent.

**The reference is RE-ARC.** A maker is written from the generator and the
verifier, its instances come from the generator, and the dataset is generator
draws — so the generator's rule is what a maker has to be right about. RE-ARC
re-implements each task rather than shipping it, and on a few tasks its rule and
the original pairs part ways on an edge case. There a maker is *supposed* to
follow the generator, and feedback demanding it match the original would be
asking it to contradict the data it is built on.

So each original pair is checked against the verifier too, and a pair the
verifier cannot reproduce either is left out of the maker's score and recorded
at severity `low` — below `critique_to_feedback.py`'s default `--min_severity`,
so it never reaches the next round. The remaining pairs still catch what this
check is for: a `generate()` that has drifted off its task.

| flag | default | |
|---|---|---|
| `--subfolder` | `arc-agi-1` | |
| `--arc_dir` | `$SOLAR_ARC_DIR`, else `./ARC-AGI/data/training` | a clone of [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) |
| `--samples` | 2 | fresh instances replayed per original pair |
| `--tasks` | all | |
| `--out` | `probe_originals.json` | |

### `critique_makers_llm.py`

Rolls out one episode and hands a separate critic the code, the trajectory and
measured static facts. The critic returns `PASS` / `REVISE` / `FAIL` plus
findings in a fixed schema. This is the check for "right answer, bad process":
the class that used to need a human reading each maker.

| flag | default | |
|---|---|---|
| `--subfolder` | `arc-agi-1` | |
| `--parallel` | 2 | |
| `--out` | — | verdicts are printed unless a path is given |
| `--dry_run` | off | |
| `--dump_payloads` | off | |
| `--save_log` | off | |

Needs the `claude` CLI.

---

## Refine

### `critique_to_feedback.py`

Turns critic verdicts into a `--task_feedback_file` for the next generation
round, so regeneration gets per-task review nobody hand-wrote.

| flag | default | |
|---|---|---|
| `--critique` | required | a `critique_makers_llm.py` output |
| `--out` | — | printed unless a path is given |
| `--min_severity` | `medium` | |
| `--verdicts` | `FAIL REVISE` | which verdicts to carry forward |
| `--forbid_ops` | off | add explicit op bans to the feedback |
| `--print_tasks` | off | list the affected task ids and exit |

Closing the loop:

```bash
python pipeline/critique_makers_llm.py  --subfolder arc-agi-1 --out critique.json
python pipeline/critique_to_feedback.py --critique critique.json --out feedback.json
python pipeline/gen_rearc_makers_llm.py --task_feedback_file feedback.json \
    --output_subdir arc-agi-1-r2 --tasks <ids> --overwrite
```

---

## Roll out

### `gen_rearc_trajectories_v2.py`

Executes each maker in ARCLE and records the episode. CPU only, no LLM calls.

| flag | default | |
|---|---|---|
| `--subfolder` | `arc-agi-1` | |
| `--num_samples` | 10 | episodes per task |
| `--num_examples` | 3 | worked examples per episode |
| `--max_grid_dim` | `30 30` | |
| `--force_grid_size` | off | see below |
| `--data_folder` | `$SOLAR_DATA_ROOT/rollout` | output root; episodes land in `<data_folder>/whole` |
| `--tasks` | all | |
| `--only_failures` | off | retry just the tasks that dropped samples |
| `--v1` | off | omit the `object_states` fields (older recording format) |
| `--demo_trajectories` | off | also record a trajectory for each worked example |

**Recording shape.** An episode of N actions has **N+1 states**: only
`operation` and `operation_name` have length N. Every tensor is padded to
`--max_grid_dim` with fill value **10** (colours occupy 0-9), and the true
extent is stored separately in `grid_dim`, `clip_dim`, `ex_in_grid_dim` and
`ex_out_grid_dim`.

**`--force_grid_size`.** Without it the ceiling is applied by *discarding*
samples that overshoot. With it the ceiling is passed into the maker so it
generates within bounds instead. At tight ceilings this is the difference
between a set and nothing, because rejection alone throws away almost
everything. The `handcraft` makers unpack `max_grid_dim` from their kwargs and
raise `KeyError` without the flag.

**`--demo_trajectories`.** Each sample fixes its worked examples and then makes
each pair in turn the target, so the demonstration pairs carry routes of their
own. The example set stays fixed across the family, which means a demonstration
target does see its own pair among its examples. That is deliberate: the point
is to show the example pairs are solvable step by step. It is not a
leave-one-out few-shot split and should not be used as one.

A sample reaches disk only if its final grid matches the target. A rollout
drops what it cannot solve rather than repairing it.

---

## Export

### `export_release.py`

Packs trajectories into parquet shards. Arrays are stored as raw
little-endian bytes with a companion `<col>__shape` column: `u1` for the pixel
planes, `i2` elsewhere. That is what makes 4,000 episodes fit in a few MB.

| flag | default | |
|---|---|---|
| `--data_root` | `$SOLAR_DATA_ROOT`, else `./solar-data` | where the rollout folders are |
| `--out` | `<data_root>/release` | |
| `--subsets` | all | `arc_agi1`, `handcraft` |
| `--shard_rows` | 500 | |
| `--verify` | 0 | round-trip this many rows per subset against the source JSON |
| `--maker_version` | off | stamp rows with the per-task generation round from `best_manifest.json` |

The exporter refuses to pack a subset whose states and actions are misaligned,
whose episodes end in a duplicated `Submit`, or where any task has fewer than
its expected episode count. A regression in the rollout cannot reach a release
quietly.

### `build_preview.py`

A second, small config whose columns are pictures rather than byte blobs, so
opening the dataset on the Hub shows grids instead of binary. One row per task.

| flag | default | |
|---|---|---|
| `--root` | required | a rollout directory, e.g. `<data_root>/<draw>/whole` |
| `--out` | required | |
| `--max_steps` | 12 | frames in the filmstrip |
| `--exclude` | — | task ids to leave out, matching `export_release.py`'s exclude |

---

## Not a script

`utils.py` holds the recording format and selection helpers. It is imported by
the rollout, not run.
