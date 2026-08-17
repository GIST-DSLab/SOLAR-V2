# SOLAR(Synethesized Offline Learning data for Abstraction and Reasoning)

**Executable ARC solutions.** Every task in this repository ships a
`grid_maker.py` that samples fresh input grids *and* emits the
[ARCLE](https://github.com/ConfeitoHS/arcle) operation sequence that solves
them — so rolling one out yields a replayable trajectory (select → recolor →
move → paste → submit), not just an input/output pair.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![ARCLE](https://img.shields.io/badge/arcle-0.2.5-orange.svg)](https://github.com/ConfeitoHS/arcle)
[![Makers](https://img.shields.io/badge/makers-433-brightgreen.svg)](#repository-layout)
[![Original pairs](https://img.shields.io/badge/original%20ARC%20pairs-1718%2F1718-brightgreen.svg)](#what-is-being-claimed)

### [Browse the trajectories &rarr;](https://qazyunho.github.io/SOLAR-V2/) &nbsp;·&nbsp; [Dataset on Hugging Face &rarr;](https://huggingface.co/datasets/dbsgh797210/SOLAR)

![one episode, action by action](figure/teaser.gif)

**400 makers, one per task of the ARC-AGI-1 training split.** Inputs come from
each task's [RE-ARC](https://github.com/michaelhodel/re-arc) generator rather
than the original pairs, so any maker can be rolled out for as many fresh
instances as you want — the published dataset is one draw of them, not their
limit (see [Data](#data)). Two smaller sets sit beside them:
`maker/arc-1d` (18) and `maker/handcraft` (15).

---

## What is being claimed

Getting the right answer is cheap to check and cheap to fake — a maker can copy
the target grid into place and call it a solution. The property this pipeline
exists to enforce is harder: **the trajectory must be one a policy could have
produced without seeing the answer.** It reads the rule from the worked
examples and executes it, rather than reconstructing a memorized output.

That judgement used to cost a human per task. Here it is made by a
generate → check → refine loop, and the makers in this repo are its output.

## Prerequisites

| | |
|---|---|
| Python | 3.11 (developed and verified on 3.11.15) |
| Dependencies | `pip install -r requirements.txt` |
| ARC-AGI-1 data | Only for `pipeline/probe_originals.py`. A clone of [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI); point at it with `--arc_dir` or `$SOLAR_ARC_DIR` |
| `claude` CLI | Only for the two LLM scripts (`pipeline/gen_rearc_makers_llm.py`, `pipeline/critique_makers_llm.py`). They shell out to [Claude Code](https://claude.com/claude-code) — no separate API key |

`requirements.txt` pins `arcle == 0.2.5`; the recording format depends on that
version's observation dict.

## Installation

```bash
git clone https://github.com/QAZyunho/SOLAR-V2.git solar-traj
cd solar-traj
pip install -r requirements.txt
```

No build step and no package install — every entry point is a script in the
repository root, run from the repository root.

## Where rollouts are written

Rollouts do not live in the repository; a full draw is tens of gigabytes. The
scripts that read or write them take a directory, and default to `./solar-data`:

```bash
export SOLAR_DATA_ROOT=/somewhere/with/space          # or pass --data_root
export SOLAR_ARC_DIR=/path/to/ARC-AGI/data/training   # probe_originals.py only
```

`--data_root`, `--out`, `--arc_dir` and `--data_folder` each override their
environment variable.

## Quickstart

```bash
# 1. roll a maker set out into trajectories  (CPU, a few minutes)
python pipeline/gen_rearc_trajectories_v2.py --subfolder arc-agi-1 --num_samples 10 \
    --rand_seed 0 --max_grid_dim 30 30 --data_folder /tmp/traj

# 2. look at one
python viz/make_teaser.py     --root /tmp/traj/whole --task 05f2a901 --out figure/mine.png
python viz/make_teaser_gif.py --root /tmp/traj/whole --task 05f2a901 --out figure/mine.gif
streamlit run viz/viz_trajectories.py

# 3. check a maker set without any LLM calls
python pipeline/probe_originals.py --subfolder arc-agi-1
```

A sample reaches disk only if its final grid matches the target, so a rollout
**drops** what it cannot solve rather than repairing it. The released draw kept
3,975 of 4,000, with every task yielding at least 6.

One flag differs by maker set: the `handcraft` makers read `max_grid_dim` out of
their kwargs and fail without `--force_grid_size`, while the LLM-written sets run
either way. The released dataset was rolled out without it.

## Repository layout

```
.
├── docs/                             the GitHub Pages viewer; carries no data of its own
│   ├── .nojekyll
│   ├── arcle_reference.md
│   ├── hero.png
│   └── index.html
├── figure/                           teaser.png and teaser.gif
│   ├── teaser.gif
│   └── teaser.png
├── maker/                            keep at the root — makers assume it
│   ├── arc-1d/                        18  1D-ARC task families, LLM-written
│   │   └── … 18 task directories
│   ├── arc-agi-1/                    400  ARC-AGI-1 training tasks, LLM-written
│   │   └── … 400 task directories
│   ├── handcraft/                     15  hand-written, same tasks as arc-agi-1
│   │   └── … 15 task directories
│   ├── __init__.py
│   ├── base_grid_maker.py
│   └── sel_helpers.py
├── pipeline/                         GENERATE / CHECK / REFINE, plus rollout and export
│   ├── build_preview.py
│   ├── critique_makers_llm.py
│   ├── critique_to_feedback.py
│   ├── export_release.py
│   ├── gen_rearc_makers_llm.py
│   ├── gen_rearc_trajectories_v2.py
│   ├── probe_originals.py
│   ├── utils.py
│   └── verify_grid_makers.py
├── re-arc/                           vendored RE-ARC generators, verifiers, DSL (Apache-2.0)
│   ├── NOTICE
│   ├── dsl.py
│   ├── generators.py
│   ├── utils.py
│   └── verifiers.py
├── viz/                              figures, GIF, gallery pages, Streamlit viewer
│   ├── build_handcraft_gallery.py
│   ├── build_hero_bg.py
│   ├── make_teaser.py
│   ├── make_teaser_gif.py
│   └── viz_trajectories.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

`maker/` and `re-arc/` cannot be moved. Every generated `grid_maker.py`
resolves the repo root as its own `parents[3]` and puts `<root>/re-arc` on
`sys.path` itself, so relocating either breaks 433 files that are outputs, not
sources. Everything else is ours and moves freely.

The 15 `handcraft` makers cover tasks `arc-agi-1` also covers, written by hand
before this pipeline existed. They are kept as the honest control: the same
task, solved by a person and by a model, in the same action space.

## Command reference

Defaults below are read from each script's `argparse` block.

### Pipeline

| Script | Role | Key flags (default) |
|---|---|---|
| `pipeline/gen_rearc_makers_llm.py` | **GENERATE** — an LLM writes a maker per task; the simulation check runs inside the same conversation | `--output_subdir` (`arc-from-rearc`), `--tasks`, `--num_examples` (6), `--parallel` (4), `--attempts` (3), `--trajectory_mode` (`efficient`), `--task_feedback_file`, `--overwrite`, `--dry_run` |
| `pipeline/verify_grid_makers.py` | **CHECK** — do N fresh samples all reach the target | `--subfolder` (`arc-agi-1`), `--num_samples`, `--tasks`, `--show_fail` |
| `pipeline/critique_makers_llm.py` | **CHECK** — an LLM replays an episode and judges whether the route is honest | `--subfolder` (`arc-agi-1`), `--parallel` (2), `--out`, `--dry_run`, `--dump_payloads`, `--save_log` |
| `pipeline/probe_originals.py` | **CHECK** — replay the solution on the task's original ARC pairs | `--subfolder` (`arc-agi-1`), `--arc_dir`, `--samples`, `--out` |
| `pipeline/critique_to_feedback.py` | **REFINE** — turn verdicts into a `--task_feedback_file` | `--critique` (required), `--out`, `--min_severity` (`medium`), `--verdicts` (`FAIL REVISE`), `--forbid_ops`, `--print_tasks` |

`pipeline/verify_grid_makers.py` reports three gates per task: **A** trajectory
correctness (ops replayed in ARCLE reach the target), **B** example correctness
(`derive_operations` also works on each worked example), **C** learnability (the
test output is inferable from the examples).

### Rollout and release

| Script | Role | Key flags (default) |
|---|---|---|
| `pipeline/gen_rearc_trajectories_v2.py` | Roll makers out into trajectories | `--subfolder` (`arc-agi-1`), `--num_samples` (10), `--num_examples` (3), `--max_grid_dim` (`30 30`), `--force_grid_size`, `--data_folder`, `--tasks`, `--v1`, `--only_failures` |
| `pipeline/export_release.py` | Pack trajectories into parquet shards | `--data_root`, `--out`, `--subsets` (all three), `--shard_rows`, `--verify`, `--maker_version` |
| `viz/make_teaser.py` / `viz/make_teaser_gif.py` | Render one trajectory as a figure / GIF | `--root`, `--task` (required), `--out`, `--max_steps`, `--ms`, `--hold`, `--dpi` |
| `viz/build_handcraft_gallery.py` | one page per task with every hand-written variant on adjacent rows | `--root`, `--out`, `--variants` (`expert half`) |
| `viz/build_hero_bg.py` | the overview page's backdrop, a mosaic of real grids | `--preview`, `--out`, `--width`, `--height`, `--seed` |
| `viz/viz_trajectories.py` | Streamlit viewer — run with `streamlit run` | — |
| `pipeline/utils.py` | Recording format and selection helpers (imported, not run) | — |

### Grid size control

`pipeline/gen_rearc_trajectories_v2.py` pads every recorded tensor to `--max_grid_dim`
with fill value **10** (colors occupy 0–9), and stores the true extent
separately in `grid_dim` / `clip_dim` / `ex_in_grid_dim` / `ex_out_grid_dim`.

Without `--force_grid_size` the ceiling is applied by *discarding* samples that
overshoot it. With the flag, the ceiling is passed into the maker so it
generates within bounds instead — which matters at tight ceilings, where
rejection alone throws away nearly everything.

## How a maker is made

A maker starts as one LLM generation — `generate()`, `sample_colors()` and
`derive_operations()` for a single task, written from the RE-ARC generator and
verifier plus that task's original ARC pairs. It then has to survive **four
kinds of check**:

| check | what it asks | where |
|---|---|---|
| **simulation** | do the ops turn I into O; does the grid revisit a state; is any op removable with O still reached | inside the generating conversation, so a rejection is re-prompted immediately, up to 3 tries |
| **samples** | do N *fresh* instances all reach the target, not just the ones it was written on | `pipeline/verify_grid_makers.py` |
| **critic** | replaying an episode, does the route match the solver's concept | `pipeline/critique_makers_llm.py` |
| **originals** | does the same solution replay on the task's own original ARC pairs | `pipeline/probe_originals.py` |

**These are four filters, not four stages.** They were added at different points
and run in different orders and combinations from round to round — the honest
description is that every maker ended up passing all four, not that it walked a
fixed 1→2→3→4. Whatever a round found,
`pipeline/critique_to_feedback.py` turned into per-task feedback and
regeneration started again with it in the prompt. The loop ran for several rounds
and each task kept its best maker. Nothing here was generated in one shot.

The feedback that closed the hardest cases carried no diagnosis — just the
failing original pair, what the trajectory produced instead, and a couple of
instances the maker's own `generate()` makes. Naming the cause is the model's
job. A human writing *"handle the other mirror axis too"* once per task is the
bottleneck this pipeline removes.

## Writing your own maker

A maker subclasses `BaseGridMaker` and implements `parse(**kwargs)`, returning
`(ex_in, ex_out, pr_in, pr_out, desc)` per sample, where `desc` carries
`operations` and `selections`. Start from any file under `maker/arc-agi-1/`,
then:

```bash
python pipeline/probe_originals.py     --subfolder <your_set>   # solves the original pairs?
python pipeline/verify_grid_makers.py  --subfolder <your_set>   # fresh samples reach the target?
python pipeline/critique_makers_llm.py --subfolder <your_set>   # is the trajectory honest?
```

Run `pipeline/probe_originals.py` first while iterating: no LLM calls, 400 tasks in
minutes, and it is the check that catches a `generate()` that has drifted from
the task. Pipe any output through `pipeline/critique_to_feedback.py` and back into
`pipeline/gen_rearc_makers_llm.py --task_feedback_file` to close the loop.

`docs/arcle_reference.md` is the operation reference makers are written
against, and is also what the generator prompt is built from.

## Data

The rolled-out trajectories are published as a separate dataset in parquet,
with the loader and schema documented there:
**[dbsgh797210/SOLAR](https://huggingface.co/datasets/dbsgh797210/SOLAR)**.

```python
from datasets import load_dataset
ds = load_dataset("dbsgh797210/SOLAR", "arc_agi1", split="train")
```

Nothing here depends on it. `pipeline/gen_rearc_trajectories_v2.py` regenerates
trajectories from the makers on CPU in a few minutes.

## License and attribution

Apache-2.0 (see [LICENSE](LICENSE)). Built on, and would not exist without:

- [RE-ARC](https://github.com/michaelhodel/re-arc) (Michael Hodel, Apache-2.0) —
  the input generators and verifiers vendored in `re-arc/` (see
  [`re-arc/NOTICE`](re-arc/NOTICE))
- [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) (Apache-2.0) — the tasks
- [1D-ARC](https://github.com/khalil-research/1D-ARC) — the 1-D task families
- [ARCLE](https://github.com/ConfeitoHS/arcle) — the environment these
  trajectories are written in
- [SOLAR-Generator](https://github.com/GIST-DSLab/SOLAR-Generator) (GIST-DSLab) —
  the trajectory recording format this builds on
