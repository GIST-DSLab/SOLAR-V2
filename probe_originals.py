#!/usr/bin/env python3
"""Stage 4 — does the maker's solution work on the ORIGINAL ARC pairs?

Stages 1-3 all judge a maker against grids its own generate() produced. That is
circular: a maker whose generate() samples a narrower slice of the task than the
authors' pairs passes every one of them and still cannot solve the task. Measured
on arc-best: 26 of 400 tasks fail at least one original pair, and the critic's
confidence does not predict which — so this is an independent signal, not a
restatement of stage 3.

Output is critique_makers_llm.py's record schema, so critique_to_feedback.py
consumes it unchanged and the finding reaches regeneration by the existing path.

The evidence is deliberately raw — the failing original pair, what the trajectory
actually produced, and samples from the maker's own generate(). It does not say
which of generate() or derive_operations is wrong, or why. Naming the cause is
the model's job; a human writing "handle the other mirror axis too" here is the
per-task hand-holding this pipeline exists to remove.

    python probe_originals.py --subfolder arc-best --out probe_originals.json
    python critique_to_feedback.py --critique probe_originals.json --out fb.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from arcle.loaders import Loader

SOLAR_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils  # noqa: E402

MAX_GRID_DIM = (30, 30)
DEFAULT_ARC = Path("/hdd_data/yunho/ARC-AGI/data/training")

# Same schema as critique_makers_llm.py's FINDING_CODES. Both codes are about the
# same observation; they differ in who is known to be wrong, which is decided by
# whether the vendored verifier can reproduce the original pairs.
FINDING_CODES = {
    "FAILS_ORIGINAL_PAIR":
        "the trajectory does not reproduce the task's own original ARC pairs, "
        "though the vendored verifier does",
    "UPSTREAM_PAIR_UNVERIFIED":
        "the trajectory does not reproduce some original ARC pairs, and neither "
        "does the vendored verifier — the upstream model of this task is itself "
        "incomplete",
}


def grid_str(g) -> str:
    return "\n".join("".join(str(int(c)) for c in row) for row in np.asarray(g))


def load_module(path: Path):
    uniq = f"probe_{path.parent.parent.name}_{path.parent.name}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(uniq, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class OriginalPairLoader(Loader):
    """Feed ARCLE the authors' pairs, one problem per pair."""

    def __init__(self, pairs, examples):
        self.pairs, self.examples = pairs, examples
        self._pathlist = [""]
        self.data = self.parse()

    def get_path(self, **kwargs):
        return [""]

    def parse(self, **kwargs):
        ei = [np.array(p["input"], dtype=np.uint8) for p in self.examples]
        eo = [np.array(p["output"], dtype=np.uint8) for p in self.examples]
        return [(ei, eo, [np.array(p["input"], dtype=np.uint8)],
                 [np.array(p["output"], dtype=np.uint8)], {"id": str(i)})
                for i, p in enumerate(self.pairs)]


def verifier_reproduces(task: str, pairs: list, rearc_root: Path) -> tuple[int, int]:
    """How many original pairs the vendored RE-ARC verifier gets right."""
    rs = str(rearc_root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    try:
        import verifiers as V
    except Exception:
        return (-1, len(pairs))
    fn = getattr(V, f"verify_{task}", None)
    if fn is None:
        return (-1, len(pairs))
    ok = 0
    for p in pairs:
        try:
            got = fn(tuple(tuple(int(x) for x in row) for row in p["input"]))
            ok += [list(r) for r in got] == p["output"]
        except Exception:
            pass
    return (ok, len(pairs))


def generate_samples(mod, n: int) -> list:
    """A few instances of what this maker's own generate() produces."""
    try:
        gm = mod.GridMaker(rand_seed=0, num_samples=n, num_examples=3)
    except Exception:
        return []
    out = []
    for _, _, pr_in, pr_out, _ in list(gm.data)[:n]:
        out.append((np.asarray(pr_in[0]), np.asarray(pr_out[0])))
    return out


def probe(task: str, maker_path: Path, arc_dir: Path, n_samples: int,
          rearc_root: Path) -> dict:
    raw = json.loads((arc_dir / f"{task}.json").read_text())
    pairs = raw["train"] + raw["test"]
    rec = {"task_id": task, "verdict": "PASS", "findings": [],
           "original_pairs": {"ok": 0, "total": len(pairs)}}

    try:
        mod = load_module(maker_path)
    except Exception as e:
        rec["error"] = f"import: {type(e).__name__}: {e}"
        return rec
    derive = getattr(mod, "derive_operations", None)
    if derive is None:
        rec["error"] = "no module-level derive_operations"
        return rec

    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=OriginalPairLoader(pairs, raw["train"][:3]),
                   max_grid_size=MAX_GRID_DIM, colors=10,
                   max_episode_steps=None, max_trial=1)
    ok, failures = 0, []
    try:
        for i, p in enumerate(pairs):
            I, O = np.array(p["input"], int), np.array(p["output"], int)
            try:
                ops, sels = derive(I.tolist(), O.tolist())
            except Exception as e:
                failures.append((i, I, O, None, [], f"{type(e).__name__}: {e}"))
                continue
            names = [solar_utils.mapping_operation(int(o)) for o in ops]
            try:
                obs, _ = env.reset(options={"prob_index": i, "adaptation": False})
                for op, sel in zip(ops, sels):
                    mask = solar_utils.to_sel_mask(sel, MAX_GRID_DIM)
                    obs, _, _, _, _ = env.step(
                        {"selection": mask.astype(bool), "operation": int(op)})
            except Exception as e:
                failures.append((i, I, O, None, names, f"{type(e).__name__}: {e}"))
                continue
            h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
            got = np.asarray(obs["grid"])[:h, :w].astype(int)
            if got.shape == O.shape and np.array_equal(got, O):
                ok += 1
            else:
                failures.append((i, I, O, got, names, ""))
    finally:
        env.close()

    rec["original_pairs"]["ok"] = ok
    if not failures:
        return rec

    v_ok, v_tot = verifier_reproduces(task, pairs, rearc_root)
    code = ("UPSTREAM_PAIR_UNVERIFIED" if 0 <= v_ok < v_tot
            else "FAILS_ORIGINAL_PAIR")
    rec["verdict"] = "FAIL" if ok == 0 else "REVISE"

    i, I, O, got, names, err = failures[0]
    ev = [f"Your solution was replayed on this task's {v_tot} original ARC pairs "
          f"(the ones the task ships with, not instances your generate() produced). "
          f"It reproduced {ok} of {v_tot}."]
    if v_ok >= 0:
        ev.append(f"The RE-ARC verifier reproduces {v_ok}/{v_tot} of the same pairs.")
    ev += ["", f"First failing original pair (#{i + 1}):", "input:", grid_str(I),
           "expected output:", grid_str(O)]
    if got is not None:
        ev += ["what your operations produced:", grid_str(got)]
    if names:
        ev.append("operations: " + " -> ".join(names))
    if err:
        ev.append(f"error: {err}")

    samples = generate_samples(mod, n_samples)
    if samples:
        ev += ["", f"For comparison, {len(samples)} instances your own generate() "
               f"produces for this task:"]
        for si, (gi, go) in enumerate(samples, 1):
            ev += [f"sample {si} input:", grid_str(gi),
                   f"sample {si} output:", grid_str(go)]

    rec["findings"].append({
        "code": code,
        "severity": "high" if ok == 0 else "medium",
        "evidence": "\n".join(ev),
        "failed_pairs": [f[0] for f in failures],
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="arc-best")
    ap.add_argument("--arc_dir", default=str(DEFAULT_ARC))
    ap.add_argument("--rearc_root", default=str(SOLAR_ROOT / "re-arc"))
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--samples", type=int, default=2,
                    help="instances of the maker's own generate() to attach as evidence")
    ap.add_argument("--out", default="probe_originals.json")
    args = ap.parse_args()

    base = SOLAR_ROOT / "maker" / args.subfolder
    arc = Path(args.arc_dir)
    tasks = sorted(p.name for p in base.iterdir()
                   if p.is_dir() and (arc / f"{p.name}.json").is_file())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    recs, n_ok, n_tot, passed = [], 0, 0, 0
    for n, t in enumerate(tasks, 1):
        r = probe(t, base / t / "grid_maker.py", arc, args.samples,
                  Path(args.rearc_root))
        recs.append(r)
        op = r.get("original_pairs", {})
        n_ok += op.get("ok", 0); n_tot += op.get("total", 0)
        passed += int(r.get("verdict") == "PASS" and "error" not in r)
        if n % 50 == 0:
            print(f"  {n}/{len(tasks)}  pairs {n_ok}/{n_tot}", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=1))
    from collections import Counter
    codes = Counter(f["code"] for r in recs for f in r.get("findings", []))
    print(f"\ntasks {len(tasks)} | original pairs {n_ok}/{n_tot} "
          f"({100 * n_ok / max(n_tot, 1):.1f}%) | PASS {passed}/{len(tasks)}")
    for c, k in codes.most_common():
        print(f"  {c}: {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
