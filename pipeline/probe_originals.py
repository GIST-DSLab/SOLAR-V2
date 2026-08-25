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
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from arcle.loaders import Loader

SOLAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR_ROOT))
import utils as solar_utils  # noqa: E402
# A maker set under maker/ can be a symlink to another disk. Each maker
# resolves its own root with .resolve(), which follows that link out of the
# repository, so re-arc never reaches sys.path from there — put it on here,
# where the root is known. It goes on AFTER the import above: re-arc has a
# utils.py of its own, and this line would otherwise shadow ours.
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

MAX_GRID_DIM = (30, 30)
# A clone of fchollet/ARC-AGI. Override with --arc_dir, or set SOLAR_ARC_DIR.
DEFAULT_ARC = Path(os.environ.get(
    "SOLAR_ARC_DIR", Path.cwd() / "ARC-AGI" / "data" / "training"))

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


def verifier_reproduces(task: str, pairs: list, rearc_root: Path) -> list:
    """Per pair: does the vendored RE-ARC verifier reproduce the original output?

    RE-ARC re-implements each task, and on a few tasks its rule and the original
    disagree on an edge case — measured over the 400 tasks here, 9 of them differ
    on at least one original pair (6cf79266, for one: RE-ARC fills a 3x3 block
    the original leaves clipped at the border). A maker written against the
    generator reproduces RE-ARC's rule, so on those pairs it is *supposed* to
    differ from the original, and asking a model to fix it sends it chasing a
    contradiction. Knowing which pair is which is what keeps that out of the
    feedback. `None` means the verifier could not be consulted at all.
    """
    rs = str(rearc_root)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    try:
        import verifiers as V
    except Exception:
        return [None] * len(pairs)
    fn = getattr(V, f"verify_{task}", None)
    if fn is None:
        return [None] * len(pairs)
    out = []
    for p in pairs:
        try:
            got = fn(tuple(tuple(int(x) for x in row) for row in p["input"]))
            out.append([list(r) for r in got] == p["output"])
        except Exception:
            out.append(False)
    return out


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
           "original_pairs": {"ok": 0, "total": len(pairs), "divergent": 0}}

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

    # A pair the RE-ARC verifier cannot reproduce either is not this maker's
    # fault: the generator it was written against models the task differently
    # from the original there. Those pairs are reported, not held against it.
    vrep = verifier_reproduces(task, pairs, rearc_root)
    divergent = [f for f in failures if vrep[f[0]] is False]
    real = [f for f in failures if vrep[f[0]] is not False]
    n_div = len(divergent)
    judged = len(pairs) - n_div
    rec["original_pairs"]["divergent"] = n_div

    if not real:
        # every failure was one of those pairs — nothing to regenerate for.
        rec["findings"].append({
            "code": "UPSTREAM_PAIR_UNVERIFIED",
            "severity": "low",
            "evidence": (f"{n_div} of this task's {len(pairs)} original pairs are "
                         f"not reproduced by the RE-ARC verifier either "
                         f"(pairs {', '.join('#' + str(f[0] + 1) for f in divergent)}). "
                         f"The maker matches the generator's rule; the generator "
                         f"and the original task disagree there."),
            "failed_pairs": [f[0] for f in divergent],
        })
        return rec

    code = "FAILS_ORIGINAL_PAIR"
    rec["verdict"] = "FAIL" if ok == 0 else "REVISE"

    i, I, O, got, names, err = real[0]
    ev = [f"Your solution was replayed on this task's {judged} original ARC pairs "
          f"(the ones the task ships with, not instances your generate() produced). "
          f"It reproduced {ok} of {judged}."]
    if n_div:
        ev.append(f"({n_div} further pair(s) are excluded: the RE-ARC verifier does "
                  f"not reproduce them either, so the generator and the original "
                  f"task disagree there and nothing needs fixing for them.)")
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
        "failed_pairs": [f[0] for f in real],
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="arc-agi-1")
    ap.add_argument("--arc_dir", default=str(DEFAULT_ARC),
                    help="ARC-AGI training JSON directory "
                         "(default: $SOLAR_ARC_DIR, else ./ARC-AGI/data/training)")
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

    recs, n_ok, n_tot, n_div, passed = [], 0, 0, 0, 0
    for n, t in enumerate(tasks, 1):
        r = probe(t, base / t / "grid_maker.py", arc, args.samples,
                  Path(args.rearc_root))
        recs.append(r)
        op = r.get("original_pairs", {})
        n_ok += op.get("ok", 0); n_tot += op.get("total", 0)
        n_div += op.get("divergent", 0)
        passed += int(r.get("verdict") == "PASS" and "error" not in r)
        if n % 50 == 0:
            print(f"  {n}/{len(tasks)}  pairs {n_ok}/{n_tot}", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=1))
    from collections import Counter
    codes = Counter(f["code"] for r in recs for f in r.get("findings", []))
    judged = n_tot - n_div
    print(f"\ntasks {len(tasks)} | original pairs {n_ok}/{judged} "
          f"({100 * n_ok / max(judged, 1):.1f}%) | PASS {passed}/{len(tasks)}")
    if n_div:
        print(f"  {n_div} pair(s) excluded: the RE-ARC verifier does not "
              f"reproduce them either")
    for c, k in codes.most_common():
        print(f"  {c}: {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
