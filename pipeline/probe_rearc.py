#!/usr/bin/env python3
"""Does the maker solve the generator's own instances?

Under `gen_rearc_trajectories_v2.py --rearc_generate --verify_filter` the data
does not come from the maker's `generate()` at all: instances are drawn from
RE-ARC's `generate_<task>`, capped to the grid size we asked for, and kept only
when `verify_<task>` reproduces them. All the maker supplies is
`derive_operations` — so that is the only thing left to be right or wrong.

A maker whose `generate()` samples a narrow slice can carry a
`derive_operations` that only handles that slice, pass every check built on its
own instances, and still fail here. This script is that gap, measured: it draws
instances the same way the rollout does and replays the maker's ops on them.

No LLM calls. Output is critique_makers_llm.py's record schema, so
critique_to_feedback.py turns it into per-task feedback unchanged. The evidence
is raw — the instance, the output the verifier vouches for, and what the ops
produced instead.

    python pipeline/probe_rearc.py --subfolder arc-agi-1 --out probe_rearc.json
    python pipeline/critique_to_feedback.py --critique probe_rearc.json --out fb.json
"""
from __future__ import annotations

import argparse
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
# A maker set under maker/ can be a symlink to another disk; see probe_originals.
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

MAX_GRID_DIM = (30, 30)

FINDING_CODES = {
    "FAILS_REARC_INSTANCE":
        "derive_operations does not solve instances drawn from the task's own "
        "RE-ARC generator and vouched for by its verifier",
}


def grid_str(g) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def totuple(g):
    return tuple(tuple(int(x) for x in row) for row in np.asarray(g))


class _PairLoader(Loader):
    """One problem per drawn instance, with fixed worked examples."""

    def __init__(self, pairs, examples):
        self.pairs, self.examples = pairs, examples
        self._pathlist = [""]
        self.data = self.parse()

    def get_path(self, **kwargs):
        return [""]

    def parse(self, **kwargs):
        ei = [np.asarray(i, dtype=np.uint8) for i, _ in self.examples]
        eo = [np.asarray(o, dtype=np.uint8) for _, o in self.examples]
        return [(ei, eo, [np.asarray(i, dtype=np.uint8)],
                 [np.asarray(o, dtype=np.uint8)], {"id": str(k)})
                for k, (i, o) in enumerate(self.pairs)]


_REARC = {}


def _rearc():
    """RE-ARC's own generators and verifiers, with its own `utils` under them.

    generators.py needs re-arc's utils (unifint and friends), but this package
    has a utils.py too and it is already in sys.modules under that name — so a
    plain `import generators` binds the wrong one and every generate_* call dies
    on a NameError that the draw loop reads as a rejected instance. Force the
    re-arc copies in, the way a generated maker does.
    """
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


def draw(task: str, n: int, max_hw, retries: int = 120):
    """Instances exactly as the rollout draws them: generator, size cap, verifier."""
    G, V = _rearc()
    gen = getattr(G, f"generate_{task.split('-')[0]}", None)
    ver = getattr(V, f"verify_{task.split('-')[0]}", None)
    if gen is None or ver is None:
        return None
    Hc, Wc = max_hw
    out = []
    for _ in range(n * retries):
        if len(out) >= n:
            break
        lb = random.random() * 0.8
        try:
            d = gen(lb, min(1.0, lb + 0.3))
        except Exception:
            continue
        I, O = np.asarray(d["input"], int), np.asarray(d["output"], int)
        if max(I.shape[0], O.shape[0]) > Hc or max(I.shape[1], O.shape[1]) > Wc:
            continue
        try:
            if totuple(ver(totuple(I))) != totuple(O):
                continue
        except Exception:
            continue
        out.append((I, O))
    return out


def probe(task: str, maker_path: Path, n: int, n_examples: int, seed: int) -> dict:
    rec = {"task_id": task, "verdict": "PASS", "findings": [],
           "instances": {"ok": 0, "total": 0}}
    random.seed(seed)
    np.random.seed(seed)
    drawn = draw(task, n + n_examples, MAX_GRID_DIM)
    if drawn is None:
        rec["error"] = "no generator or verifier for this task"
        return rec
    if len(drawn) < n + n_examples:
        rec["error"] = f"generator yielded {len(drawn)} verified instances, need {n + n_examples}"
        return rec

    examples, pairs = drawn[:n_examples], drawn[n_examples:]
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

    env = gym.make("ARCLE/O2ARCv2Env-v0", render_mode=None,
                   data_loader=_PairLoader(pairs, examples),
                   max_grid_size=MAX_GRID_DIM, colors=10,
                   max_episode_steps=None, max_trial=1)
    ok, failures = 0, []
    try:
        for i, (I, O) in enumerate(pairs):
            try:
                ops, sels = derive(I.tolist(), O.tolist())
            except Exception as e:
                failures.append((I, O, None, [], f"{type(e).__name__}: {e}"))
                continue
            names = [solar_utils.mapping_operation(int(o)) for o in ops]
            try:
                obs, _ = env.reset(options={"prob_index": i, "adaptation": False})
                for op, sel in zip(ops, sels):
                    mask = solar_utils.to_sel_mask(sel, MAX_GRID_DIM)
                    obs, _, _, _, _ = env.step(
                        {"selection": mask.astype(bool), "operation": int(op)})
            except Exception as e:
                failures.append((I, O, None, names, f"{type(e).__name__}: {e}"))
                continue
            h, w = int(obs["grid_dim"][0]), int(obs["grid_dim"][1])
            got = np.asarray(obs["grid"])[:h, :w].astype(int)
            if got.shape == O.shape and np.array_equal(got, O):
                ok += 1
            else:
                failures.append((I, O, got, names, ""))
    finally:
        env.close()

    rec["instances"] = {"ok": ok, "total": len(pairs)}
    if not failures:
        return rec

    rec["verdict"] = "FAIL" if ok == 0 else "REVISE"
    I, O, got, names, err = failures[0]
    ev = [f"Your derive_operations was replayed on {len(pairs)} instances drawn "
          f"from this task's own RE-ARC generator — not from your generate() — "
          f"each one confirmed by the task's verifier. It solved {ok} of them.",
          "", "First instance it did not solve — input:", grid_str(I),
          "the output the verifier confirms for that input:", grid_str(O)]
    if got is not None:
        ev += ["what your operations produced:", grid_str(got)]
    if names:
        ev.append("operations: " + " -> ".join(names))
    if err:
        ev.append(f"error: {err}")

    rec["findings"].append({
        "code": "FAILS_REARC_INSTANCE",
        "severity": "high" if ok == 0 else "medium",
        "evidence": "\n".join(ev),
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="arc-agi-1")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--num_samples", type=int, default=5,
                    help="instances replayed per task")
    ap.add_argument("--num_examples", type=int, default=3)
    ap.add_argument("--rand_seed", type=int, default=0)
    ap.add_argument("--out", default="probe_rearc.json")
    args = ap.parse_args()

    base = SOLAR_ROOT / "maker" / args.subfolder
    tasks = sorted(p.name for p in base.iterdir()
                   if (p / "grid_maker.py").is_file())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    recs, n_ok, n_tot, passed = [], 0, 0, 0
    for i, t in enumerate(tasks, 1):
        r = probe(t, base / t / "grid_maker.py", args.num_samples,
                  args.num_examples, args.rand_seed)
        recs.append(r)
        n_ok += r["instances"]["ok"]
        n_tot += r["instances"]["total"]
        passed += int(r["verdict"] == "PASS" and "error" not in r)
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}  instances {n_ok}/{n_tot}", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=1))
    print(f"\ntasks {len(tasks)} | generator instances solved {n_ok}/{n_tot} "
          f"({100 * n_ok / max(n_tot, 1):.1f}%) | PASS {passed}/{len(tasks)}")
    errs = [r["task_id"] for r in recs if "error" in r]
    if errs:
        print(f"  not evaluated ({len(errs)}): {' '.join(errs[:12])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
