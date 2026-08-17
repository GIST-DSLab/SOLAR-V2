#!/usr/bin/env python3
"""
V2 trajectory generator: same as gen_rearc_trajectories.py but records the full
O2ARC object state at every step.

New fields per step (beyond v1):
  selected       : obs['selected']                   — env selection mask AFTER op
  object_active  : obs['object_states']['active']    — 1 if object mode active
  object         : obs['object_states']['object']    — extracted object pixels
  object_sel     : obs['object_states']['object_sel']— object shape mask
  object_dim     : obs['object_states']['object_dim']— (h, w) of object
  object_pos     : obs['object_states']['object_pos']— (r, c) top-left of object
  background     : obs['object_states']['background']— grid with object removed

This format supports Move/Rotate/Flip trajectories where the object position
changes between steps, so the model can track the object across operations.

Usage:
    python gen_rearc_trajectories_v2.py --subfolder arc-from-rearc-v2 \\
        --num_samples 100 --data_folder ../ARC_Single_llm_v2_obj
"""
import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from tqdm import tqdm

# The repo keeps this in pipeline/ and the working tree at its root, so a fixed
# parents[N] is wrong in one of them — copying the file between the two put
# MAKER_BASE outside the tree. Find the ancestor that actually holds maker/.
SOLAR_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "maker").is_dir())
sys.path.insert(0, str(SOLAR_ROOT))

import utils as solar_utils

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--num_samples",   type=int, default=10)
parser.add_argument("--num_examples",  type=int, default=3)
parser.add_argument("--rand_seed",     type=int, default=0)
parser.add_argument("--max_grid_dim",  nargs=2, type=int, default=[30, 30])
parser.add_argument("--force_grid_size", action="store_true", default=False,
                    help="Ask the maker to generate within --max_grid_dim instead of "
                         "discarding samples that overshoot it. Needs a maker built "
                         "from a gen_rearc_makers.py that supports max_grid_dim.")
parser.add_argument("--data_folder",   type=str,
                    default=str(SOLAR_ROOT.parent / "ARC_Single_llm_v2_obj"))
parser.add_argument("--subfolder",     type=str, default="arc-agi-1")
parser.add_argument("--skip_on_error", action="store_true", default=True)
parser.add_argument("--v1", action="store_true", default=False,
                    help="Output v1-compatible format (no object_states fields).")
parser.add_argument("--tasks", nargs="+", default=None,
                    help="Only regenerate these task IDs (e.g. --tasks 045e512c 025d127b).")
parser.add_argument("--only_failures", action="store_true", default=False,
                    help="Save ONLY samples whose output is wrong (for inspecting failures).")
parser.add_argument("--demo_trajectories", action="store_true", default=False,
                    help="Also emit a full trajectory for every demonstration example, "
                         "not just the problem pair. Each maker-sample of N demos + 1 "
                         "problem is expanded into N+1 targets sharing a group_id, with "
                         "the N demonstrations fixed as the worked examples for every "
                         "target; each row is tagged role=problem|demo. Opt-in; off "
                         "leaves output byte-identical to before.")
args = parser.parse_args()

MAX_GRID_DIM = tuple(args.max_grid_dim)
H, W = MAX_GRID_DIM
formatted_time = datetime.now().strftime("%y.%m.%d")

MAKER_BASE = SOLAR_ROOT / "maker" / args.subfolder
task_dirs  = sorted(MAKER_BASE.iterdir())
if args.tasks:
    task_dirs = [d for d in task_dirs if d.name in args.tasks]

whole_root = Path(args.data_folder) / "whole"
whole_root.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pad(arr: np.ndarray, fill: int = 10) -> np.ndarray:
    """Pad any (H', W') ndarray to MAX_GRID_DIM with fill value."""
    out = np.full(MAX_GRID_DIM, fill, dtype=np.int8)
    h, w = min(arr.shape[0], H), min(arr.shape[1], W)
    out[:h, :w] = arr[:h, :w]
    return out


def _pad_bool(arr: np.ndarray) -> np.ndarray:
    """Pad a boolean/uint8 selection mask to MAX_GRID_DIM with zeros."""
    out = np.zeros(MAX_GRID_DIM, dtype=np.int8)
    h, w = min(arr.shape[0], H), min(arr.shape[1], W)
    out[:h, :w] = arr[:h, :w]
    return out


def _record_step(data: dict, obs: dict, action_sel_bbox, action_op: int,
                 reward: int, term: bool, step: int, v2: bool = True) -> None:
    """Append one step worth of state to data dict."""
    grid_template = np.full(MAX_GRID_DIM, 10, dtype=np.int8)

    gd_h, gd_w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
    c_h,  c_w  = int(obs["clip_dim"][0]), int(obs["clip_dim"][1])

    grid_pad = grid_template.copy()
    grid_pad[:gd_h, :gd_w] = obs["grid"][:gd_h, :gd_w]

    clip_pad = grid_template.copy()
    if c_h > 0 and c_w > 0:
        clip_pad[:c_h, :c_w] = obs["clip"][:c_h, :c_w].astype(np.int8)

    data["grid"].append(grid_pad.tolist())
    data["grid_dim"].append([gd_h, gd_w])
    data["clip"].append(clip_pad.tolist())
    data["clip_dim"].append([int(c_h), int(c_w)])
    _sel_mask = solar_utils.to_sel_mask(action_sel_bbox, MAX_GRID_DIM)
    data["selection"].append(solar_utils.mask_to_bbox(_sel_mask))
    data["selection_mask"].append(_sel_mask.tolist())
    data["operation"].append(action_op)
    data["operation_name"].append(solar_utils.mapping_operation(action_op))
    data["reward"].append(reward)
    data["terminated"].append(term)
    data["step"].append(step)

    if not v2:
        return

    # v2 fields — object state from env obs
    obj = obs["object_states"]
    data["selected"].append(_pad_bool(obs["selected"]).tolist())
    data["object_active"].append(int(obj["active"]))
    data["object"].append(_pad(obj["object"], fill=0).tolist())
    data["object_sel"].append(_pad_bool(obj["object_sel"]).tolist())
    data["object_dim"].append([int(obj["object_dim"][0]), int(obj["object_dim"][1])])
    data["object_pos"].append([int(obj["object_pos"][0]), int(obj["object_pos"][1])])
    data["background"].append(_pad(obj["background"], fill=0).tolist())


def _expand_demo_targets(loader, gm_mod) -> None:
    """Fixed-demonstration expansion for --demo_trajectories.

    Each maker-sample of N demos + 1 problem is rewritten into N+1 loader.data
    entries — every pair takes a turn as the "problem" (the reset target ARCLE
    replays), while the worked examples stay FIXED to the sample's N
    demonstrations for every target (the test pair is never shown as a
    demonstration). The problem reuses its precomputed ops; each demo derives
    fresh ops via the maker's own
    `derive_operations`. group_id/role/example_index ride in `desc`. The env is
    built AFTER this, so prob_index addresses every target. A demo whose derive
    raises is dropped here; one whose replay later mismatches is dropped by the
    existing validation in run_task — neither drops its siblings.
    """
    import inspect

    derive = gm_mod.derive_operations
    n_params = len(inspect.signature(derive).parameters)

    new_data = []
    for sample in loader.data:
        ex_in_list, ex_out_list, pr_in_list, pr_out_list, desc = sample
        if not pr_in_list or not pr_out_list:
            new_data.append(sample)          # degenerate sample: leave untouched
            continue

        base_id = str(desc["id"])
        concept = desc.get("concept", "")
        pairs = list(zip(ex_in_list, ex_out_list)) + [(pr_in_list[0], pr_out_list[0])]
        prob_idx = len(pairs) - 1

        for k, (Ik, Ok) in enumerate(pairs):
            if k == prob_idx:
                ops, sels = desc["operations"], desc["selections"]
                row_id, role = base_id, "problem"
            else:
                try:
                    ops, sels = derive(Ik, Ok) if n_params >= 2 else derive(Ik)
                except Exception:
                    continue                 # drop this demo target, keep siblings
                row_id, role = f"{base_id}_ex{k}", "demo"

            # Worked examples are the sample's FIXED demonstrations — the same
            # set for every target in the family, so the demonstration panel is
            # identical across all N+1 episodes. The problem (test) pair is never
            # shown as a demonstration; a demo target still sees the full demo
            # set (itself included) rather than a leave-one-out subset.
            ex_i = list(ex_in_list)
            ex_o = list(ex_out_list)
            new_data.append((ex_i, ex_o, [Ik], [Ok], {
                "id":            row_id,
                "concept":       concept,
                "operations":    ops,
                "selections":    sels,
                "group_id":      base_id,
                "role":          role,
                "example_index": k,
            }))

    loader.data = new_data


def run_task(tid: str, maker_path: Path) -> tuple[int, int]:
    """Generate trajectories for one task. Returns (correct, total)."""

    spec = importlib.util.spec_from_file_location(f"gm_{tid}", str(maker_path))
    gm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm_mod)
    GridMaker = gm_mod.GridMaker

    loader = GridMaker(
        rand_seed=args.rand_seed,
        num_samples=args.num_samples,
        num_examples=args.num_examples,
        # Makers that understand it generate within the ceiling instead of
        # letting the oversize filter below throw the sample away. Omit the
        # kwarg entirely when the flag is off — passing None instead makes every
        # maker that unpacks it into (max_h, max_w) raise on the first sample.
        **({"max_grid_dim": MAX_GRID_DIM} if args.force_grid_size else {}),
    )

    # Expand demos into their own targets BEFORE the env is built, so prob_index
    # addresses every one of them.
    if args.demo_trajectories:
        _expand_demo_targets(loader, gm_mod)

    env = gym.make(
        "ARCLE/O2ARCv2Env-v0",
        render_mode=None,
        data_loader=loader,
        max_grid_size=MAX_GRID_DIM,
        colors=10,
        max_episode_steps=None,
        max_trial=1,
    )

    folder = whole_root / f"test.{tid}.s{H}.{formatted_time}"
    folder.mkdir(exist_ok=True)

    correct = total = 0

    for i, sample in enumerate(loader.data):
        ex_in_list, ex_out_list, pr_in_list, pr_out_list, desc = sample
        if not pr_in_list or not pr_out_list:
            continue

        ops  = desc["operations"]
        sels = desc["selections"]
        pr_out = pr_out_list[0]
        g_h, g_w = pr_out.shape

        # skip samples whose grid exceeds MAX_GRID_DIM
        if g_h > H or g_w > W:
            continue

        try:
            obs, info = env.reset(options={"prob_index": i, "adaptation": False})
        except Exception as e:
            tqdm.write(f"  SKIP {tid}[{i}] reset: {e}")
            continue

        grid_template = np.full(MAX_GRID_DIM, 10, dtype=np.int8)

        _desc_out = {"id": desc["id"], "concept": desc.get("concept", "")}
        # Provenance for --demo_trajectories rides in desc so the round-trip
        # exporter (which ignores desc) needs no special-casing. Absent when the
        # flag is off, leaving the record byte-identical to before.
        for _extra in ("group_id", "role", "example_index"):
            if _extra in desc:
                _desc_out[_extra] = desc[_extra]

        data = {
            "desc":         _desc_out,
            "step":         [],
            "selection":    [],
            "operation":    [],
            "operation_name": [],
            "reward":       [],
            "terminated":   [],
            "grid_dim":     [],
            "grid":         [],
            "clip_dim":     [],
            "clip":         [],
            "selection_mask": [],
            "in_grid":      None,
            "out_grid":     None,
            "ex_in":        [],
            "ex_out":       [],
            "ex_in_grid_dim":  [],
            "ex_out_grid_dim": [],
        }
        if not args.v1:
            data.update({
                "selected":      [],
                "object_active": [],
                "object":        [],
                "object_sel":    [],
                "object_dim":    [],
                "object_pos":    [],
                "background":    [],
            })

        v2 = not args.v1

        # step 0: initial state (no action yet)
        _record_step(data, obs, sels[0], ops[0], 0, False, 0, v2=v2)

        # step 1..N: execute each op
        step_err = False
        for s, (op, sel) in enumerate(zip(ops, sels)):
            sel_mask = solar_utils.to_sel_mask(sel, MAX_GRID_DIM)
            action   = {"selection": sel_mask.astype(bool), "operation": op}
            try:
                obs, reward, term, trunc, info = env.step(action)
            except Exception as e:
                tqdm.write(f"  SKIP {tid}[{i}] step {s} op={op}: {e}")
                step_err = True
                break

            next_sel = sels[s + 1] if s + 1 < len(sels) else sel
            next_op  = ops[s + 1]  if s + 1 < len(ops)  else op
            _record_step(data, obs, next_sel, next_op, int(reward), bool(term), s + 1, v2=v2)

        if step_err:
            continue

        # Drop the terminal filler action: the last recorded step is the post-Submit
        # terminal STATE (it keeps its grid, reward and terminated=True), but it has no
        # action to take from it. Recording one made every trajectory end in Submit,Submit.
        # Result: operation/operation_name have length N (the real actions, ending in a
        # single Submit) while the state arrays stay N+1 — the standard RL alignment.
        data["operation"] = data["operation"][:-1]
        data["operation_name"] = data["operation_name"][:-1]

        data["in_grid"]  = data["grid"][0]
        data["out_grid"] = data["grid"][-1]

        for ei, eo in zip(ex_in_list, ex_out_list):
            solar_utils.append_example(data, ei, eo, grid_template.copy())

        # validate
        final = np.array(data["grid"][-1])[:g_h, :g_w]
        ok = np.array_equal(final.astype(np.uint8), pr_out)
        total += 1
        correct += ok

        if not ok:
            tqdm.write(f"  FAIL {tid}[{i}]: ops={ops[:4]}")
            if not args.only_failures:
                continue
        elif args.only_failures:
            continue                       # only_failures mode: keep just the wrong ones

        data = solar_utils.convert_npint_to_int(data)
        # The filename comes from the maker's own id, and nothing obliges a maker
        # to vary it per sample: the handcraft set returns a constant, so 25
        # samples all wrote one path and 24 were silently lost. Only the last
        # survived, and the run still reported 375/375 correct. Disambiguate on
        # collision rather than trusting the id to be unique.
        stem = str(desc["id"])
        path = folder / f"{stem}.json"
        if path.exists():
            k = 1
            while (folder / f"{stem}_{k}.json").exists():
                k += 1
            path = folder / f"{stem}_{k}.json"
        with open(path, "w") as f:
            json.dump(data, f)

    return correct, total


# ── main loop ─────────────────────────────────────────────────────────────────
total_correct = total_total = 0
failed_tasks = []

for task_dir in tqdm(task_dirs, desc="tasks"):
    maker_path = task_dir / "grid_maker.py"
    if not maker_path.exists():
        continue
    tid = task_dir.name
    try:
        c, t = run_task(tid, maker_path)
        total_correct += c
        total_total   += t
        if c < t:
            failed_tasks.append((tid, c, t))
    except Exception as e:
        failed_tasks.append((tid, "ERR", str(e)[:80]))
        if not args.skip_on_error:
            raise
        tqdm.write(f"ERROR {tid}: {e}")

print(f"\nDone: {total_correct}/{total_total} correct "
      f"({100*total_correct/max(total_total,1):.1f}%)")
print(f"Output: {whole_root}")
if failed_tasks:
    print(f"Failed ({len(failed_tasks)}):")
    for f in failed_tasks:
        print(" ", f)
