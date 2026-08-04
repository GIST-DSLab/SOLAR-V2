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
[![Original pairs](https://img.shields.io/badge/original%20ARC%20pairs-1718%2F1718-brightgreen.svg)](#verified-status)

![one episode, action by action](figure/teaser.gif)

418 makers were written by an LLM (Claude Opus) and filtered by the pipeline in
this repo; 15 more were written by hand as a control. The trajectories they
produce are published separately as a dataset (see [Data](#data)).

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
| ARC-AGI-1 data | Only for `pipeline/probe_originals.py`. Defaults to `/hdd_data/yunho/ARC-AGI/data/training`; override with `--arc_dir` |
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
├── docs/                             the operation space makers are written against
│   └── arcle_reference_v2.md
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
├── viz/                              figure, GIF, Streamlit viewer
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
| `pipeline/gen_rearc_makers_llm.py` | **GENERATE** — an LLM writes a maker per task; the simulation check runs inside the same conversation | `--output_subdir` (`arc-from-rearc`), `--tasks`, `--num_examples` (6), `--parallel` (4), `--attempts` (3), `--prompt_version` (`v1`\|`v2`\|`v3`), `--task_feedback_file`, `--overwrite`, `--dry_run` |
| `pipeline/verify_grid_makers.py` | **CHECK** — do N fresh samples all reach the target | `--subfolder` (`arc-from-rearc-v6`), `--num_samples`, `--tasks`, `--show_fail` |
| `pipeline/critique_makers_llm.py` | **CHECK** — an LLM replays an episode and judges whether the route is honest | `--subfolder` (`arc-from-rearc-v6`), `--parallel` (2), `--out`, `--dry_run`, `--dump_payloads`, `--save_log` |
| `pipeline/probe_originals.py` | **CHECK** — replay the solution on the task's original ARC pairs | `--subfolder` (`arc-best`), `--arc_dir`, `--samples`, `--out` |
| `pipeline/critique_to_feedback.py` | **REFINE** — turn verdicts into a `--task_feedback_file` | `--critique` (required), `--out`, `--min_severity` (`medium`), `--verdicts` (`FAIL REVISE`), `--forbid_ops`, `--print_tasks` |

`pipeline/verify_grid_makers.py` reports three gates per task: **A** trajectory
correctness (ops replayed in ARCLE reach the target), **B** example correctness
(`derive_operations` also works on each worked example), **C** learnability (the
test output is inferable from the examples).

### Rollout and release

| Script | Role | Key flags (default) |
|---|---|---|
| `pipeline/gen_rearc_trajectories_v2.py` | Roll makers out into trajectories | `--subfolder` (`arc-from-rearc-v2`), `--num_samples` (10), `--num_examples` (3), `--max_grid_dim` (`30 30`), `--force_grid_size`, `--data_folder`, `--tasks`, `--v1`, `--only_failures` |
| `pipeline/export_release.py` | Pack trajectories into parquet shards | `--out`, `--subsets` (`arc_agi1` `arc_1d`), `--shard_rows`, `--verify`, `--maker_version` |
| `viz/make_teaser.py` / `viz/make_teaser_gif.py` | Render one trajectory as a figure / GIF | `--root`, `--task` (required), `--out`, `--max_steps`, `--ms`, `--hold`, `--dpi` |
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

```
  GENERATE   an LLM writes generate() / sample_colors() / derive_operations()
             for one task, from the RE-ARC generator and verifier plus the
             task's original ARC pairs
      │
      ▼
  CHECK      simulation   do the ops turn I into O; does the grid revisit a
                          state; is any op removable with O still reached —
                          inside the generating conversation, so a rejection is
                          re-prompted immediately, up to 3 tries
             samples      pipeline/verify_grid_makers.py — N fresh instances must all
                          reach the target, not just the ones it was written on
             critic       pipeline/critique_makers_llm.py — an LLM replays an episode and
                          judges the route against the solver's concept
             originals    pipeline/probe_originals.py — the same solution replayed on the
                          task's own original ARC pairs
      │
      ▼
  REFINE     pipeline/critique_to_feedback.py turns every finding into per-task feedback,
             and regeneration starts again from GENERATE with it in the prompt
```

The loop ran for several rounds and each task kept its best maker. Nothing here
was generated in one shot.

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

`docs/arcle_reference_v2.md` is the operation reference makers are written
against, and is also what the generator prompt is built from.

## Known gaps

Found by running each entry point from a clean checkout. None affect the
makers or `pipeline/probe_originals.py`; all are in the tooling around them.

- **`pipeline/gen_rearc_makers_llm.py` does not start.** It reads `arcle_reference.md`
  and `arcle_reference_v2.md` from the repository root
  ([line 39–40](pipeline/gen_rearc_makers_llm.py)), but the repo ships only
  `docs/arcle_reference_v2.md`, so importing it raises `FileNotFoundError`.
  Copy or symlink both files to the root to run it.

- **Check scripts default to maker sets that are not in this repo.**
  `--subfolder` defaults to `arc-from-rearc-v6` (`pipeline/verify_grid_makers.py`,
  `pipeline/critique_makers_llm.py`) and `arc-best` (`pipeline/probe_originals.py`). Always pass
  `--subfolder arc-agi-1`, `arc-1d`, or `handcraft` explicitly.

- **`pipeline/verify_grid_makers.py` errors on makers that emit cell-list selections.**
  Line 203 unpacks a selection as a 4-tuple bbox (`r2, c2, sh2, sw2 = sel2`),
  but some makers emit `{'cells': [[r, c], ...]}`. The task aborts with
  `ERROR: not enough values to unpack (expected 4, got 1)`. Observed on 8 of 25
  sampled `arc-agi-1` tasks and on `1d_fill`. Only the B2 sub-check needs the
  bbox form; A and C are unaffected. `pipeline/probe_originals.py` does not have this
  limitation.

- **Absolute paths baked into `pipeline/export_release.py` and `pipeline/probe_originals.py`.**
  Both resolve data under `/hdd_data/yunho`. `pipeline/probe_originals.py` exposes
  `--arc_dir`; `pipeline/export_release.py`'s `DATA_ROOT` and per-subset `root` are
  module constants that must be edited.

- **`pipeline/export_release.py` references assets not in this repo** — maker sets
  `maker/arc-best`, `maker/arc-agi2-solve`, `maker/arc-agi2-construction`, and
  the script `gen_agi2_llm.py` (quoted in its `rollout` strings). Only the
  `arc_1d` subset maps onto a maker set present here.

- **No CI.** There is no `.github/` directory; the badges above are static.

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
