#!/usr/bin/env python3
"""Do the instances a maker produces obey the generator's rule?

Every other check asks whether the trajectory reaches the maker's own target.
This one asks whether that target is the right one: the maker is written from a
RE-ARC generator, its instances are drawn from it, and the released dataset is
those draws — so the generator's rule, as the vendored verifier implements it,
is what a maker has to be right about. A `generate()` that drifts produces
perfectly self-consistent instances of a task nobody asked for, and nothing else
in the pipeline notices.

No LLM calls. Output is critique_makers_llm.py's record schema, so
critique_to_feedback.py turns it into per-task feedback unchanged.

The evidence is raw: the input the maker made, the output it claims, and the
output the verifier gives for that same input. It does not say whether
generate() wandered or derive_operations solved the wrong thing.

    python pipeline/probe_generator.py --subfolder arc-agi-1 --out probe_gen.json
    python pipeline/critique_to_feedback.py --critique probe_gen.json --out fb.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

SOLAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLAR_ROOT))
# A maker set under maker/ can be a symlink to another disk; see probe_originals.
sys.path.insert(0, str(SOLAR_ROOT / "re-arc"))

MAX_GRID_DIM = (30, 30)

FINDING_CODES = {
    "GENERATOR_RULE_MISMATCH":
        "the instances generate() produces do not follow the rule the task's "
        "RE-ARC generator implements — the verifier maps the same input to a "
        "different output",
}


def grid_str(g) -> str:
    return "\n".join("".join(str(int(v)) for v in row) for row in np.asarray(g))


def totuple(g):
    return tuple(tuple(int(x) for x in row) for row in np.asarray(g))


def load_maker(path: Path):
    spec = importlib.util.spec_from_file_location(f"gm_{path.parent.name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(task: str, maker_path: Path, seeds: list[int], n: int) -> dict:
    # A variant id (`<task>-half`) carries the base task's rule.
    base = task.split("-")[0]
    rec = {"task_id": task, "verdict": "PASS", "findings": [],
           "instances": {"ok": 0, "total": 0}}
    try:
        import verifiers as V
    except Exception as e:
        rec["error"] = f"verifiers: {type(e).__name__}: {e}"
        return rec
    verify = getattr(V, f"verify_{base}", None)
    if verify is None:
        rec["error"] = "no verifier for this task"
        return rec
    try:
        GridMaker = load_maker(maker_path).GridMaker
    except Exception as e:
        rec["error"] = f"import: {type(e).__name__}: {e}"
        return rec

    ok, bad = 0, []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        try:
            loader = GridMaker(rand_seed=seed, num_samples=n, num_examples=3)
            loader.parse(num_samples=n, num_examples=3,
                         max_grid_dim=list(MAX_GRID_DIM))
        except Exception as e:
            bad.append((None, None, None, f"{type(e).__name__}: {e}"))
            continue
        for _ex_in, _ex_out, pr_in, pr_out, _desc in loader.data:
            if not pr_in or not pr_out:
                continue
            I, O = np.asarray(pr_in[0], int), np.asarray(pr_out[0], int)
            try:
                got = np.asarray(verify(totuple(I)), int)
            except Exception as e:
                bad.append((I, O, None, f"verifier: {type(e).__name__}: {e}"))
                continue
            if got.shape == O.shape and np.array_equal(got, O):
                ok += 1
            else:
                bad.append((I, O, got, ""))

    rec["instances"] = {"ok": ok, "total": ok + len(bad)}
    if not bad:
        return rec

    rec["verdict"] = "FAIL" if ok == 0 else "REVISE"
    I, O, got, err = bad[0]
    ev = [f"Your generate() was sampled {ok + len(bad)} times and the task's "
          f"RE-ARC verifier agreed with {ok} of them. The generator is the "
          f"reference: its rule is the one this task means."]
    if I is not None:
        ev += ["", "One instance your generate() produced — input:", grid_str(I),
               "the output your generate() paired with it:", grid_str(O)]
    if got is not None:
        ev += ["what the RE-ARC verifier maps that same input to:", grid_str(got)]
    if err:
        ev.append(f"error: {err}")

    rec["findings"].append({
        "code": "GENERATOR_RULE_MISMATCH",
        "severity": "high" if ok == 0 else "medium",
        "evidence": "\n".join(ev),
    })
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", default="arc-agi-1")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--rand_seed", type=int, nargs="+", default=[0, 1],
                    help="one or more seeds; each is an independent draw")
    ap.add_argument("--num_samples", type=int, default=2,
                    help="instances per seed")
    ap.add_argument("--out", default="probe_generator.json")
    args = ap.parse_args()

    base = SOLAR_ROOT / "maker" / args.subfolder
    tasks = sorted(p.name for p in base.iterdir()
                   if (p / "grid_maker.py").is_file())
    if args.tasks:
        tasks = [t for t in tasks if t in args.tasks]

    recs, n_ok, n_tot, passed = [], 0, 0, 0
    for i, t in enumerate(tasks, 1):
        r = probe(t, base / t / "grid_maker.py", args.rand_seed, args.num_samples)
        recs.append(r)
        n_ok += r["instances"]["ok"]
        n_tot += r["instances"]["total"]
        passed += int(r["verdict"] == "PASS" and "error" not in r)
        if i % 50 == 0:
            print(f"  {i}/{len(tasks)}  instances {n_ok}/{n_tot}", flush=True)

    Path(args.out).write_text(json.dumps(recs, indent=1))
    print(f"\ntasks {len(tasks)} | instances {n_ok}/{n_tot} "
          f"({100 * n_ok / max(n_tot, 1):.1f}%) | PASS {passed}/{len(tasks)}")
    errs = [r["task_id"] for r in recs if "error" in r]
    if errs:
        print(f"  not evaluated ({len(errs)}): {' '.join(errs[:12])}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
