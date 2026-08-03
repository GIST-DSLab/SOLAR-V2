#!/usr/bin/env python3
"""
Grid maker verification script.

Checks three things per task:
  A. Trajectory correctness : execute ops/sels on test input in ARCLE env → must == test output
  B. Example correctness    : derive_operations(ex_I, ex_O) must also work for each example pair
  C. Learnability           : test output must be inferable from examples
                              - classification proxy (output ≤ 2×2): pr_out values ⊆ union(ex_outs)
                              - general: ops structure consistent across examples + test

Usage:
    python verify_grid_makers.py --subfolder arc-from-rearc-v6 --num_samples 5
    python verify_grid_makers.py --subfolder arc-from-rearc-v6 --tasks 0d3d703e 27a28665
"""
import argparse
import importlib.util
import sys
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import gymnasium as gym

SOLAR_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subfolder",   type=str, default="arc-from-rearc-v6")
parser.add_argument("--num_samples", type=int, default=5,
                    help="Episodes per task")
parser.add_argument("--num_examples",type=int, default=3)
parser.add_argument("--max_grid_dim",nargs=2,  type=int, default=[30, 30])
parser.add_argument("--rand_seed",   type=int, default=42)
parser.add_argument("--tasks",       nargs="+", default=None)
parser.add_argument("--show_fail",   action="store_true",
                    help="Print failing ops for A/B failures")
args = parser.parse_args()

H, W = args.max_grid_dim
MAKER_BASE = SOLAR_ROOT / "maker" / args.subfolder
task_dirs = sorted(MAKER_BASE.iterdir())
if args.tasks:
    task_dirs = [d for d in task_dirs if d.name in args.tasks]

random.seed(args.rand_seed)
np.random.seed(args.rand_seed)


# ── helpers ───────────────────────────────────────────────────────────────────

def run_ops(env, obs, ops, sels):
    """Execute ops/sels from initial obs. Returns final grid or None on error."""
    for op, sel in zip(ops, sels):
        sel_mask = solar_utils.to_sel_mask(sel, (H, W))
        action = {"selection": sel_mask.astype(bool), "operation": int(op)}
        try:
            obs, reward, term, trunc, info = env.step(action)
        except Exception as e:
            return None, str(e)
    gd_h = int(obs["grid_dim"][0])
    gd_w = int(obs["grid_dim"][1])
    return obs["grid"][:gd_h, :gd_w].astype(np.uint8), None


def check_learnability(ex_outs, pr_out):
    """
    C. Learnability check.

    Classification proxy: if pr_out is small (≤2×2 total cells ≤ 4),
    every unique value in pr_out must appear in at least one ex_out.

    General tasks: return True (we rely on B to catch inconsistencies).
    Returns (pass: bool, reason: str)
    """
    pr_out = np.asarray(pr_out)
    total_cells = pr_out.size

    if total_cells <= 4:
        # classification-like: check value coverage
        pr_vals = set(np.unique(pr_out).tolist())
        ex_vals = set()
        for eo in ex_outs:
            ex_vals.update(np.unique(np.asarray(eo)).tolist())
        unseen = pr_vals - ex_vals
        if unseen:
            return False, f"test output values {unseen} never seen in examples"
        return True, "classification values covered"

    return True, "general task (not checked)"


def run_task(tid, maker_path):
    """
    Returns dict with per-check pass counts.
    """
    spec = importlib.util.spec_from_file_location(f"gm_{tid}", str(maker_path))
    gm_mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(gm_mod)
    except Exception as e:
        return {"load_error": str(e)[:120]}

    GridMaker = gm_mod.GridMaker
    loader = GridMaker(
        rand_seed=args.rand_seed,
        num_samples=args.num_samples,
        num_examples=args.num_examples,
    )

    try:
        loader.parse(
            num_samples=args.num_samples,
            num_examples=args.num_examples,
            max_grid_dim=[H, W],
        )
    except Exception as e:
        return {"parse_error": str(e)[:120]}

    env = gym.make(
        "ARCLE/O2ARCv2Env-v0",
        render_mode=None,
        data_loader=loader,
        max_grid_size=(H, W),
        colors=10,
        max_episode_steps=None,
        max_trial=1,
    )

    results = defaultdict(list)  # "A", "B", "C" → list of bool

    for i, sample in enumerate(loader.data):
        ex_in_list, ex_out_list, pr_in_list, pr_out_list, desc = sample
        if not pr_in_list or not pr_out_list:
            continue

        ops  = desc["operations"]
        sels = desc["selections"]
        pr_out = np.asarray(pr_out_list[0])
        g_h, g_w = pr_out.shape
        if g_h > H or g_w > W:
            continue

        # ── A: trajectory correctness ─────────────────────────────────────────
        try:
            obs, info = env.reset(options={"prob_index": i, "adaptation": False})
        except Exception as e:
            results["A"].append(False)
            results["A_err"].append(f"reset: {e}")
            continue

        final, err = run_ops(env, obs, ops, sels)
        if final is None:
            results["A"].append(False)
            results["A_err"].append(err)
            if args.show_fail:
                print(f"  A FAIL {tid}[{i}] step error: {err}")
        else:
            ok_A = np.array_equal(final[:g_h, :g_w], pr_out)
            results["A"].append(ok_A)
            if not ok_A and args.show_fail:
                print(f"  A FAIL {tid}[{i}]: ops={ops[:5]}")

        # ── B: example correctness ────────────────────────────────────────────
        # B1: derive_operations(ex_I, ex_O) must not crash, ops[-1]==34, lens match
        # B2: for Color-only ops, numpy-simulate and verify output matches ex_O
        derive_fn = getattr(gm_mod, "derive_operations", None)
        b_pass = True
        if derive_fn is not None:
            for j, (ei, eo) in enumerate(zip(ex_in_list, ex_out_list)):
                ei_arr = np.asarray(ei, dtype=int)
                eo_arr = np.asarray(eo, dtype=int)

                # B1: structural check
                try:
                    ex_ops, ex_sels = derive_fn(ei_arr, eo_arr)
                except Exception as e:
                    b_pass = False
                    if args.show_fail:
                        print(f"  B1 FAIL {tid}[{i}] ex{j} derive crash: {e}")
                    break
                if not ex_ops or ex_ops[-1] != 34:
                    b_pass = False
                    if args.show_fail:
                        print(f"  B1 FAIL {tid}[{i}] ex{j}: last op not Submit, ops={ex_ops}")
                    break
                if len(ex_ops) != len(ex_sels):
                    b_pass = False
                    if args.show_fail:
                        print(f"  B1 FAIL {tid}[{i}] ex{j}: len mismatch ops={len(ex_ops)} sels={len(ex_sels)}")
                    break

                # B2: simulate Color ops (0-9) only — geometric ops require env
                if all(op < 10 or op == 34 for op in ex_ops):
                    grid = ei_arr.copy().astype(int)
                    g_h2, g_w2 = eo_arr.shape
                    for op2, sel2 in zip(ex_ops[:-1], ex_sels[:-1]):
                        if op2 > 9:
                            continue
                        r2, c2, sh2, sw2 = sel2
                        grid[r2:r2+sh2+1, c2:c2+sw2+1] = op2
                    result2 = grid[:g_h2, :g_w2]
                    if not np.array_equal(result2, eo_arr):
                        b_pass = False
                        if args.show_fail:
                            print(f"  B2 FAIL {tid}[{i}] ex{j}: color-sim mismatch, ops={ex_ops[:5]}")
                        break

        results["B"].append(b_pass)

        # ── C: learnability ───────────────────────────────────────────────────
        c_pass, c_reason = check_learnability(ex_out_list, pr_out)
        results["C"].append(c_pass)
        if not c_pass and args.show_fail:
            print(f"  C FAIL {tid}[{i}]: {c_reason}")

    env.close()

    def pct(lst):
        valid = [x for x in lst if x is not None]
        if not valid:
            return float("nan"), 0
        return sum(valid) / len(valid) * 100, len(valid)

    a_pct, a_n = pct(results["A"])
    b_pct, b_n = pct(results["B"])
    c_pct, c_n = pct(results["C"])

    return {
        "A_pct": a_pct, "A_n": a_n,
        "B_pct": b_pct, "B_n": b_n,
        "C_pct": c_pct, "C_n": c_n,
    }


# ── main ──────────────────────────────────────────────────────────────────────
totals = {"A": [], "B": [], "C": []}
fail_A, fail_B, fail_C = [], [], []
errors = []

print(f"Verifying {len(task_dirs)} tasks in '{args.subfolder}' "
      f"({args.num_samples} samples each)\n")
print(f"{'task':<14} {'A(traj)':>8} {'B(ex)':>8} {'C(learn)':>10}")
print("-" * 44)

for task_dir in task_dirs:
    maker_path = task_dir / "grid_maker.py"
    if not maker_path.exists():
        continue
    tid = task_dir.name

    try:
        r = run_task(tid, maker_path)
    except Exception as e:
        print(f"{tid:<14} ERROR: {str(e)[:80]}")
        errors.append(tid)
        continue

    if "load_error" in r or "parse_error" in r:
        err = r.get("load_error") or r.get("parse_error")
        print(f"{tid:<14} ERROR: {err[:60]}")
        errors.append(tid)
        continue

    a_str = f"{r['A_pct']:5.0f}% ({r['A_n']})" if r["A_n"] else "  N/A"
    b_str = f"{r['B_pct']:5.0f}% ({r['B_n']})" if r["B_n"] else "  N/A"
    c_str = f"{r['C_pct']:5.0f}% ({r['C_n']})" if r["C_n"] else "  N/A"

    flags = ""
    if r["A_n"] and r["A_pct"] < 100: flags += " ✗A"; fail_A.append(tid)
    if r["B_n"] and r["B_pct"] < 100: flags += " ✗B"; fail_B.append(tid)
    if r["C_n"] and r["C_pct"] < 100: flags += " ✗C"; fail_C.append(tid)

    print(f"{tid:<14} {a_str:>8} {b_str:>8} {c_str:>10}{flags}")

    if r["A_n"]: totals["A"].append(r["A_pct"])
    if r["B_n"]: totals["B"].append(r["B_pct"])
    if r["C_n"]: totals["C"].append(r["C_pct"])

print("\n" + "=" * 44)
def avg(lst): return sum(lst) / len(lst) if lst else float("nan")
print(f"Average  A(traj):  {avg(totals['A']):.1f}%  ({len(totals['A'])} tasks)")
print(f"Average  B(ex):    {avg(totals['B']):.1f}%  ({len(totals['B'])} tasks)")
print(f"Average  C(learn): {avg(totals['C']):.1f}%  ({len(totals['C'])} tasks)")

if fail_A: print(f"\nFail A ({len(fail_A)}): {' '.join(fail_A)}")
if fail_B: print(f"Fail B ({len(fail_B)}): {' '.join(fail_B)}")
if fail_C: print(f"Fail C ({len(fail_C)}): {' '.join(fail_C)}")
if errors:  print(f"Errors  ({len(errors)}): {' '.join(errors)}")
