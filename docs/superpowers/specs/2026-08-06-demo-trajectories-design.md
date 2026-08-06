# Demo-example trajectories (`--demo_trajectories`) — design

**Date:** 2026-08-06
**Status:** implemented (pilot). Flag `--demo_trajectories` + `arc_agi1_demo`
release subset + `gallery_demo.html` shipped. Pilot draw = 6 tasks
(017c7c7b, 46442a0e, 47c1f68c, 73251a56, 007bbfb7, 1b2d62fb), num_samples 10 →
240 rows (60 problem + 180 demo), 240/240 replay-valid, 0 demo-derive drops.
Gallery published at claude.ai/code/artifact/fee005d0-4eec-478d-a0ca-cba31ab2191a.
Remaining: scale to full 400 tasks when desired.

## Problem

When we synthesize ARCLE trajectories, each maker-sample produces `num_examples`
demonstration pairs (`ex_in`/`ex_out`) plus one problem pair (`pr_in`/`pr_out`).
Only the **problem** pair gets a trajectory (`operations`/`selections`, replayed
through ARCLE). The demonstration pairs are stored as **static input→output
grids only**.

We want the demonstration examples to also carry full trajectories, so that:

1. **Worked-example conditioning** — a policy can see *how* each demo is solved
   (the op sequence), not just its I→O mapping.
2. **Data augmentation** — each demo becomes a complete, standalone training
   trajectory.

`derive_operations` is a general function (it measures the rule from the pair),
so it applies to a demo pair exactly as to the problem pair. No LLM call is
needed; this is deterministic.

## Non-goals

- Not changing the existing default output. This is strictly opt-in.
- Not editing the 400 per-task maker files.
- Not overwriting the existing `ARC_best10_r3` draw or `arc_agi1` release subset.
  The demo-trajectory data is a **separate versioned dataset**.

## Approach: leave-one-out expansion, flat sibling rows

When the flag is on, a maker-sample of `N` demos + 1 problem (`N+1` pairs total)
is expanded into **`N+1` targets**. Each pair takes a turn as the "problem"; the
other `N` pairs are its worked examples. Because ARCLE resets to a problem via
`prob_index` into `loader.data`, we **pre-expand `loader.data`** so every target
is addressable as an ordinary problem. The existing rollout / replay / record
loop is then reused unchanged, once per target.

This keeps every trajectory stored exactly once (flat), fits the flat parquet
schema, and serves both consumers: conditioning joins siblings by `group_id`;
augmentation uses every row as a standalone sample.

### Why not the alternatives

- **Nested `ex_trajectories` in the problem record** — record size balloons ~N×,
  awkward in the flat parquet schema (arrays-of-arrays, `__shape` column blow-up),
  and duplicates data if demos are also emitted standalone. Rejected.
- **Action-only (ops+sels) per demo** — user wants each demo to be a complete
  independent sample, which needs the full state arrays. Rejected.

## Components & changes

### 1. `gen_rearc_trajectories_v2.py` (core change)

- **New flag** `--demo_trajectories` (default **off**).
- After `GridMaker` is loaded (the module is already available as `gm_mod`, so
  `gm_mod.derive_operations` is directly callable), and **before** the env is
  built, if the flag is on: replace `loader.data` with the expanded target list.
  For each original sample `(ex_in, ex_out, pr_in, pr_out, desc)`:
  - Build the ordered pair list `pairs = list(zip(ex_in, ex_out)) + [(pr_in[0], pr_out[0])]`.
  - For each index `k` in `pairs` (the target):
    - `I_k, O_k = pairs[k]`
    - ops/sels: reuse `desc["operations"]/["selections"]` when `k` is the problem
      (last index); otherwise derive fresh via `gm_mod.derive_operations` — call
      `(I_k, O_k)` or `(I_k)` per `inspect.signature` (solve makers take I only).
    - examples for this target = every *other* pair's grids (`ex_in/ex_out`,
      leave-one-out), preserving the current per-record example shape.
    - emit a `loader.data` entry `([exI...], [exO...], [I_k], [O_k], desc_k)` where
      `desc_k` carries new keys: `group_id`, `role` (`"problem"`|`"demo"`),
      `example_index` (`k`), and a unique `id`
      (`{base_id}` for the problem, `{base_id}_ex{k}` for demos).
  - `group_id = f"{task}_{sample_n}"` — deterministic, no RNG.
  - Gotcha: `loader.data` is uint8-cast in `BaseGridMaker.__init__` via
    `convert_grid_to_uint8`. The expansion runs *after* that, so the new entries
    must reuse the already-cast grids (they do) and keep grids as uint8 arrays;
    ops are plain ints and sels are int lists, matching the current `desc`.
- **Failure isolation:** deriving/replaying a demo target may fail
  (`derive_operations` raises, or the replayed grid ≠ `O_k`). The existing
  `skip_on_error` path already drops a single failing entry and continues; we
  keep that. A dropped demo does not drop its siblings. The problem row is
  primary. Count and log dropped demo targets per task.
- **Record propagation:** the per-trajectory output JSON gains `group_id`,
  `role`, `example_index` (read from `desc_k`).

### 2. `export_release.py`

- Add three top-level columns: `group_id` (string), `role` (string),
  `example_index` (int32). They live alongside `id`/`task_id`, not in the
  byte-packed array block; `row_of` reads them from the JSON `desc`.
- **New subset** `arc_agi1_demo` in `SUBSETS`, pointing at the new whole/ dir
  (see §4), so `arc_agi1` shards are untouched. `n_steps == states-1` and the
  duplicated-Submit guards apply per row unchanged.
- Older data (no demo fields) round-trips fine: default `role="problem"`,
  `example_index=null/-1`, `group_id=id` when the keys are absent, so the existing
  `arc_agi1` export still validates.

### 3. `build_gallery_html.py` / `extract_best.py` (gallery)

- `extract_best.py` gains an optional emit of `role`/`group_id` per entry.
- Gallery default shows **only `role=="problem"`**. A separate **"show demo
  trajectories"** toggle button reveals the sibling demo trajectories (grouped by
  `group_id`). The existing (problem-only) gallery is unchanged when the demo
  dataset is absent.
- This is a *separate* gallery build/version; it does not alter the currently
  published gallery.

### 4. Data locations (separate version, nothing overwritten)

- Rollout output: `/hdd_data/yunho/ARC_best10_r3_demo/whole` (new dir; `ARC_best10_r3`
  stays as-is).
- Release: `release/data/arc_agi1_demo/*.parquet` (new subset; `arc_agi1` stays).
- Gallery: a separate `*_gallery.json` + a versioned `gallery_demo.html`, or the
  same page behind the toggle — decided at gallery-build time, not blocking here.

## Rollout plan

1. **Pilot** — run `--demo_trajectories` on a small task subset (e.g. 3–5 tasks,
   `--tasks ...`) into the new dir. Verify: N+1 rows per sample, `group_id`/`role`
   correct, every demo trajectory replay-valid, drop-rate sane.
2. Inspect a few demo trajectories in the gallery behind the toggle.
3. **Scale** — full 400-task draw into the new dir; export the `arc_agi1_demo`
   subset; build the demo gallery.

Deterministic throughout (no LLM). Size grows ~(N+1)× vs the problem-only draw
(`num_examples=3` → ~4×); acceptable because it is a separate opt-in version.

## Resolved (from review)

- Gallery: a **separate page** titled **"re-arc with demo trajectory"**
  (`gallery_demo.html`), problem-only by default with a demo-trajectory toggle.
  The existing published gallery is untouched.
- Pilot: any small task subset is fine. Proceed, then upload the demo gallery.
