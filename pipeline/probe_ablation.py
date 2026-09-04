#!/usr/bin/env python3
"""Find operations a route does not need, by deleting them and replaying.

The LLM probe in `probe_device.py` asks a model whether an operation is there
for the coordinates rather than for the rule.  It reads the route as prose and
it guesses: of fourteen tasks it flagged, six held up.  Two of its findings
were contradicted by the grids themselves, and one -- a flip it called a no-op
-- turned out to decide the answer in four episodes out of six.

This asks the trajectory instead.  Delete one operation, replay the rest, and
see whether the episode still lands on its target.  An operation that every
episode can spare is provably unnecessary; there is no judgement in it and so
no false positive.  What it cannot see is the operation that is needed but for
the wrong reason -- a transpose that earns its keep only because a later crop
was written in transposed coordinates.  The two probes miss different things.

Episodes are read as the rollout wrote them, so this needs no maker: the
operations and selections are in the file.
"""
import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR.parent))

MAX_GRID_DIM = (30, 30)
TERMINAL = {34}          # Submit -- removing it is not a finding


def _crop(g, dim):
    h, w = dim
    return np.asarray([r[:w] for r in g[:h]], dtype=int)


def _replay(I, O, ops, sels):
    """Final grid of a route, or None if the environment rejects it.

    Selections are replayed from the recorded masks, not from the bounding
    boxes beside them: a colour operation often acts on a ragged set of cells,
    and rebuilding it as a rectangle silently paints the corners too.  A
    quarter of the draw failed to reproduce its own target until this used the
    masks.
    """
    import gymnasium as gym
    from choose_maker import _One
    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=_One(I, O), max_grid_size=MAX_GRID_DIM,
                   colors=10, max_episode_steps=None, max_trial=1)
    try:
        obs, _ = env.reset(options={"prob_index": 0, "adaptation": False})
        for op, sel in zip(ops, sels):
            obs, _, _, _, _ = env.step(
                {"selection": np.asarray(sel, dtype=bool), "operation": int(op)})
        h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
        return np.asarray(obs["grid"][:h, :w], dtype=int)
    except Exception:
        return None
    finally:
        env.close()


def _episode(path: Path):
    d = json.loads(path.read_text())
    I = _crop(d["in_grid"], d["grid_dim"][0])
    O = _crop(d["out_grid"], d["grid_dim"][-1])
    n = len(d["operation"])
    masks = [np.asarray(m, dtype=bool) for m in d["selection_mask"][:n]]
    return I, O, list(d["operation"]), masks, list(d["operation_name"])


def probe_task(args):
    root, task, limit = args
    hits = sorted(Path(root, "whole").glob(f"test.{task}.*"))
    if not hits:
        return {"task_id": task, "episodes": 0, "removable": []}
    files = sorted(hits[0].glob("*.json"))[:limit]
    seen, spare = 0, []
    for f in files:
        try:
            I, O, ops, sels, names = _episode(f)
        except Exception:
            continue
        if _replay(I, O, ops, sels) is None:
            continue                       # route does not even replay
        base = _replay(I, O, ops, sels)
        if not np.array_equal(base, O):
            continue                       # route does not solve; not our question
        seen += 1
        drop = []
        for j, op in enumerate(ops):
            if op in TERMINAL:
                continue
            g = _replay(I, O, ops[:j] + ops[j + 1:], sels[:j] + sels[j + 1:])
            if g is not None and np.array_equal(g, O):
                drop.append(names[j])
        spare.append(drop)
    if not seen:
        return {"task_id": task, "episodes": 0, "removable": []}
    # An operation counts only when the same operation is spare across
    # episodes.  One redundant paint in one instance is an accident of that
    # instance; the same flip going spare every time is the route's shape.
    tally = Counter(n for d in spare for n in set(d))
    return {"task_id": task, "episodes": seen,
            "episodes_with_spare": sum(1 for d in spare if d),
            "removable": [{"op": n, "episodes": c} for n, c in tally.most_common()]}


def _flagged(r, frac):
    """True when one operation is spare in at least `frac` of the episodes."""
    return bool(r["episodes"]) and any(
        d["episodes"] >= frac * r["episodes"] for d in r["removable"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="draw folder holding whole/")
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=6, help="episodes per task")
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--min_fraction", type=float, default=0.8,
                   help="flag a task when this share of episodes can spare an op")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    tasks = a.tasks or sorted({d.name.split(".")[1]
                               for d in Path(a.root, "whole").glob("test.*")})
    jobs = [(a.root, t, a.limit) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=a.parallel) as ex:
        for i, r in enumerate(ex.map(probe_task, jobs), 1):
            out.append(r)
            if _flagged(r, a.min_fraction):
                ops = ", ".join(f"{d['op']} {d['episodes']}/{r['episodes']}"
                                for d in r["removable"]
                                if d["episodes"] >= a.min_fraction * r["episodes"])
                print(f"  {r['task_id']}  {ops}",
                      flush=True)
            if i % 50 == 0:
                print(f"[{i}/{len(jobs)}]", file=sys.stderr, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    flagged = [r for r in out if _flagged(r, a.min_fraction)]
    print(f"\n{len(out)}개 중 {len(flagged)}개에 지울 수 있는 op가 있음")


if __name__ == "__main__":
    main()
