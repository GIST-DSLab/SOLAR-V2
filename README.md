# solar-traj — LLM-written grid makers for ARC, and the pipeline that vets them

Each ARC task here has a `grid_maker.py` that does two things: sample fresh
input grids for the task, and derive the sequence of
[ARCLE](https://github.com/ConfeitoHS/arcle) operations that solves them. Roll
one out and you get a replayable trajectory — select a region, recolor it, move
an object, paste, submit — not just an input/output pair.

433 of them were written by an LLM (Claude Opus) and filtered by the pipeline in
this repo. The trajectories they produce are published separately as a dataset
(see *Data*).

## What is actually being claimed

Getting the right answer is easy to check and easy to fake — a maker can copy
the target grid into place and call it a solution. The harder property, and the
one this pipeline exists to enforce, is that **the trajectory is one a policy
could have produced without seeing the answer**: it reads the rule from the
worked examples and executes it, rather than reconstructing a memorized output.

That judgement used to take a human per task, which does not scale. Here it is
made by a three-stage gate, and the makers in this repo are what survived it.

## Layout

```
maker/
  arc-agi-1/<task_id>/grid_maker.py   400  ARC-AGI-1 training tasks, LLM-written
  arc-1d/<family>/grid_maker.py        18  1D-ARC task families, LLM-written
  handcraft/<task_id>/grid_maker.py    15  hand-written, same tasks as arc-agi-1
  base_grid_maker.py  sel_helpers.py

re-arc/                    vendored RE-ARC generators/verifiers/DSL (Apache-2.0)
docs/arcle_reference_v2.md the operation space the makers are written against
figure/teaser.png

gen_rearc_makers_llm.py    generate makers with an LLM  (stage 2 lives inside)
verify_grid_makers.py      stage 1 — do N fresh samples all reach the target
critique_makers_llm.py     stage 3 — LLM critic judges the trajectory, not the answer
critique_to_feedback.py    turn findings into per-task feedback for the next round
gen_rearc_trajectories_v2.py  roll makers out into trajectories
export_release.py          pack trajectories into parquet shards
make_teaser.py             render one trajectory as a figure
viz_trajectories.py        Streamlit viewer
utils.py                   recording format and selection helpers
```

The 15 `handcraft` makers cover tasks that `arc-agi-1` also covers, written by
hand before this pipeline existed. They are kept because they are the honest
control: the same task, solved by a person and by a model, in the same action
space.

## Quickstart

```bash
pip install -r requirements.txt        # gymnasium, arcle, numpy, pyarrow, matplotlib

# roll a maker set out into trajectories
python gen_rearc_trajectories_v2.py --subfolder arc-agi-1 --num_samples 10 \
    --rand_seed 0 --max_grid_dim 30 30 --data_folder /tmp/traj

# look at one
python make_teaser.py --root /tmp/traj/whole --task 6fa7a44f --out figure/mine.png
streamlit run viz_trajectories.py
```

**`--force_grid_size` differs by maker set.** The `handcraft` makers read
`max_grid_dim` from their kwargs and fail without it, so they need the flag; the
LLM-written sets work with or without it, and the published dataset was rolled
out **without** it. Concretely, at `--num_samples 2`:

| set | command | result |
|---|---|---|
| `arc-agi-1` | as above | 794/800 samples reach the target (99.2%) |
| `arc-1d` | as above | 36/36 (100%) |
| `handcraft` | `+ --force_grid_size` | 30/30 (100%) |

A sample is written to disk only if its final grid matches the target, so the
shortfall is dropped rather than repaired.

## How a maker is made

```
generate ─► stage 2  in-process validation
            · do the ops actually turn I into O (simulated on 2–3 pairs)
            · does the grid ever return to a state it already passed through
            · drop any op and does O still appear → redundant → reject
            · the rejection reason goes back into the same conversation, up to 3 attempts
         ─► stage 1  verify_grid_makers.py — N fresh samples must all match
         ─► stage 3  LLM critic replays one episode in ARCLE and judges the
                     *trajectory* against the solver's concept, not the answer
         ─► findings become per-task reviewer feedback for the next round
```

Generation ran in rounds, each round's findings feeding the next; every task
kept the maker that came out best.

## Data

The rolled-out trajectories — 4,154 episodes in parquet, with the loader and the
schema documented — are published as a separate dataset: **TODO: link**.

Nothing here depends on that dataset; `gen_rearc_trajectories_v2.py` regenerates
trajectories from the makers on CPU in a few minutes.

## Writing your own maker

A maker subclasses `BaseGridMaker` and implements `parse(**kwargs)`, returning
sampled examples plus the operation sequence that solves each one. Start from
any file under `maker/arc-agi-1/`, then:

```bash
python verify_grid_makers.py --subfolder <your_set>       # do the samples reach the target
python critique_makers_llm.py --subfolder <your_set>      # is the trajectory honest
```

`docs/arcle_reference_v2.md` is the operation reference the makers are written
against, and is also what the generator prompt is built from.

## License and attribution

Apache-2.0. Built on, and would not exist without:

- [RE-ARC](https://github.com/michaelhodel/re-arc) (Michael Hodel, Apache-2.0) —
  the input generators and verifiers vendored in `re-arc/`
- [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) (Apache-2.0) — the tasks
- [1D-ARC](https://github.com/khalil-research/1D-ARC) — the 1-D task families
- [ARCLE](https://github.com/ConfeitoHS/arcle) — the environment these
  trajectories are written in
- [SOLAR-Generator](https://github.com/GIST-DSLab/SOLAR-Generator) (GIST-DSLab) —
  the trajectory recording format this builds on
