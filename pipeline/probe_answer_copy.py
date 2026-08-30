#!/usr/bin/env python3
"""Hand the maker the wrong answer and see whether it draws it.

`derive_operations(I, O)` is allowed to look at O — the maker is written in
construction mode. What it may not do is take the rule from O, because then the
trajectory is a transcription of an answer nobody would have in front of them.
Reading the code tells you that (the critic does), but reading is arguable and
what comes back is a verdict, not something to fix.

This makes the counterexample instead. Two instances of the same task with the
same grid shapes; call derive_operations with the input of one and the output of
the other. A route that measures its parameters from I cannot reach that other
output. A route that reads them off O reproduces it, and the trajectory it
returns is proof: those ops, on this input, produce a grid that is not the
answer to it.

No LLM calls. Output is critique_makers_llm.py's record schema.

    python pipeline/probe_answer_copy.py --subfolder arc-agi-1 --out copy.json
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from arcle.loaders import Loader

SOLAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils  # noqa: E402
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

MAX_GRID_DIM = (30, 30)

FINDING_CODES = {
    "ANSWER_COPIED_FROM_O":
        "given the output of a different instance, derive_operations produced "
        "that output — the route's parameters come from O, not from I",
}

_REARC = {}


def _rearc():
    """re-arc's generators and verifiers, with its own utils under them."""
    if _REARC:
        return _REARC["G"], _REARC["V"]
    rs = str(SOLAR_ROOT / "re-arc")
    while rs in sys.path:
        sys.path.remove(rs)
    sys.path.insert(0, rs)
    for name in ("utils", "dsl", "generators", "verifiers"):
        sys.modules.pop(name, None)
    import generators as G
    import verifiers as V
    _REARC.update(G=G, V=V)
    return G, V


def grid_str(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def totuple(g):
    return tuple(tuple(int(x) for x in row) for row in np.asarray(g))


class _One(Loader):
    def __init__(self, I, O):
        self.pair = (I, O)
        self._pathlist = [""]
        self.data = self.parse()

    def get_path(self, **k):
        return [""]

    def parse(self, **k):
        I, O = self.pair
        a, b = np.asarray(I, np.uint8), np.asarray(O, np.uint8)
        return [([a], [b], [a], [b], {"id": "0"})]


def draw(task: str, n: int, seed: int):
    G, V = _rearc()
    gen = getattr(G, f"generate_{task.split('-')[0]}", None)
    ver = getattr(V, f"verify_{task.split('-')[0]}", None)
    if gen is None or ver is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    out = []
    for _ in range(n * 60):
        if len(out) >= n:
            break
        lb = random.random() * 0.8
        try:
            d = gen(lb, min(1.0, lb + 0.3))
        except Exception:
            continue
        I, O = np.asarray(d["input"], int), np.asarray(d["output"], int)
        if max(I.shape) > MAX_GRID_DIM[0] or max(O.shape) > MAX_GRID_DIM[1]:
            continue
        try:
            if totuple(ver(totuple(I))) != totuple(O):
                continue
        except Exception:
            continue
        out.append((I, O))
    return out


def replay(derive, I, O_shown):
    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=_One(I, O_shown), max_grid_size=MAX_GRID_DIM,
                   colors=10, max_episode_steps=None, max_trial=1)
    try:
        ops, sels = derive(I.tolist(), O_shown.tolist())
        obs, _ = env.reset(options={"prob_index": 0, "adaptation": False})
        for op, sel in zip(ops, sels):
            obs, _, _, _, _ = env.step(
                {"selection": solar_utils.to_sel_mask(sel, MAX_GRID_DIM).astype(bool),
                 "operation": int(op)})
        h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
        return np.asarray(obs["grid"])[:h, :w].astype(int), ops
    except Exception:
        return None, None
    finally:
        env.close()


def agree(a, b):
    if a is None or a.shape != b.shape:
        return 0.0
    return float((a == b).mean())


def probe(task: str, maker_path: Path, trials: int, seed: int) -> dict:
    rec = {"task_id": task, "verdict": "PASS", "findings": [],
           "copied": 0, "trials": 0}
    inst = draw(task, 40, seed)
    if not inst:
        rec["error"] = "no generator or verifier for this task"
        return rec
    spec = importlib.util.spec_from_file_location(f"gm_{task}", str(maker_path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        rec["error"] = f"import: {type(e).__name__}: {e}"
        return rec
    derive = getattr(mod, "derive_operations", None)
    if derive is None:
        rec["error"] = "no module-level derive_operations"
        return rec

    # only a pair of the same shapes can be decisive: a copier has to be able to
    # land on the other output for its landing there to mean anything
    buckets = collections.defaultdict(list)
    for I, O in inst:
        buckets[(I.shape, O.shape)].append((I, O))
    pairs = []
    for group in buckets.values():
        for k in range(0, len(group) - 1, 2):
            (I1, O1), (I2, O2) = group[k], group[k + 1]
            if not np.array_equal(O1, O2):
                pairs.append((I1, O1, O2))
    if not pairs:
        rec["error"] = "no two instances share their grid shapes"
        return rec

    worst = None
    wrong_scores = []
    for I1, O1, O2 in pairs[:trials]:
        got, ops = replay(derive, I1, O2)
        rec["trials"] += 1
        w = agree(got, O2)
        wrong_scores.append(w)
        if got is not None and got.shape == O2.shape and np.array_equal(got, O2):
            rec["copied"] += 1
        if worst is None or w > worst[0]:
            worst = (w, I1, O1, O2, got, ops)

    mean_wrong = sum(wrong_scores) / len(wrong_scores)
    if rec["copied"] == 0 and mean_wrong < 0.5:
        return rec

    w, I1, O1, O2, got, ops = worst
    names = [solar_utils.mapping_operation(int(o)) for o in ops] if ops else []
    rec["verdict"] = "FAIL" if rec["copied"] else "REVISE"
    ev = [f"Your derive_operations was called with the input of one instance and "
          f"the output of a different one. It should not have been able to reach "
          f"that output — it is not the answer to this input — but what your ops "
          f"produced matches it {w*100:.0f}% cell for cell"
          + (f", exactly, on {rec['copied']} of {rec['trials']} such pairs."
             if rec["copied"] else f" (over {rec['trials']} such pairs)."),
          "", "the input you were given:", grid_str(I1),
          "the answer to that input:", grid_str(O1),
          "the output of the other instance, which is what you were shown:",
          grid_str(O2)]
    if got is not None:
        ev += ["what your operations produced from that input:", grid_str(got)]
    if names:
        ev.append("operations: " + " -> ".join(names))
    rec["findings"].append({
        "code": "ANSWER_COPIED_FROM_O",
        "severity": "high" if rec["copied"] else "medium",
        "evidence": "\n".join(ev),
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="arc-agi-1")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--trials", type=int, default=5,
                    help="mismatched pairs tried per task")
    ap.add_argument("--rand_seed", type=int, default=900)
    ap.add_argument("--out", default="probe_answer_copy.json")
    args = ap.parse_args()

    base = SOLAR_ROOT / "maker" / args.subfolder
    tasks = sorted(p.name for p in base.iterdir() if (p / "grid_maker.py").is_file())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    recs, copied, revised, err = [], 0, 0, 0
    for i, t in enumerate(tasks, 1):
        r = probe(t, base / t / "grid_maker.py", args.trials, args.rand_seed)
        recs.append(r)
        copied += r["verdict"] == "FAIL"
        revised += r["verdict"] == "REVISE"
        err += "error" in r
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=1))
    print(f"\ntasks {len(tasks)} | drew the wrong answer exactly: {copied} | "
          f"followed it closely: {revised} | not evaluated: {err}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
