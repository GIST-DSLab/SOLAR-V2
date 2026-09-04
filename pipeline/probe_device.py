#!/usr/bin/env python3
"""Is an operation there to move the coordinates, or to carry out the rule?

The checks that already run cannot see this one. The generator refuses a
candidate with an operation that can be deleted while still reaching the
target, and b8cdaf2b passes it: a FlipH sits between two paints of the same
region so the second lands on the mirrored cells, and deleting any one of the
three breaks the route. The algebra that catches a mirror and its undo cannot
see it either, because there is only one flip and one flip is not the identity.
And asking whether the output is invariant under the flip catches every task
whose rule is to build something symmetric.

What is left is a judgement, so it is put to a model with the verifier in front
of it, one route at a time. Not two routes compared: pitting them against each
other made length decide the answer, the shorter route winning on both sides
having been told not to count length. Asked separately, the four routes flagged
across eighty-six averaged five operations against fifteen for the rest, which
is the bias pointing the other way and the reason to trust the answers.

No grids are altered and nothing is regenerated here. Output is
critique_makers_llm.py's record schema, so critique_to_feedback.py carries a
finding into the next round unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOLAR_ROOT = Path(__file__).resolve().parents[1]

QUESTION = """Here is one ARC task's verifier, an input and its target, and one route through ARCLE that turns the input into the target.

The verifier is the task written in RE-ARC's DSL. The route is a list of ARCLE operations, each applied to a selected region.

Verifier:

{src}

Input:
{I}

Target:
{O}

Route:
{route}

Question: does this route contain any operation that is there to make the coordinates convenient rather than to carry out the rule?

What that looks like: a flip or rotation used so that a later selection lands somewhere else, rather than because the rule turns anything; a transform the verifier applies only to normalise its own working and then undoes; a turn whose effect is invisible because the region it acts on is symmetric. What it does not look like: an operation that performs a reflection, rotation or move the task is actually about, however many of them there are, and a long run of paints that simply colours what the rule says to colour.

Judge the route on its own. Do not consider whether it could have been shorter.

Answer with exactly one line of JSON and nothing else:
{{"device": true or false, "which": "<the operation numbers, or empty>", "why": "<one sentence, at most 25 words>"}}"""


def verifier_source(task: str, rearc_root: str) -> str | None:
    if rearc_root not in sys.path:
        sys.path.insert(0, rearc_root)
    try:
        import inspect
        import verifiers as V
        return inspect.getsource(getattr(V, f"verify_{task}"))
    except Exception:
        return None


def episode(root: str, task: str):
    hit = sorted(glob.glob(f"{root}/test.{task}.*/0.json"))
    if not hit:
        return None
    d = json.loads(Path(hit[0]).read_text())

    def crop(g, dim):
        h, w = dim
        return ["".join(str(c) for c in r[:w]) for r in g[:h]]

    return {"ops": d["operation_name"],
            "I": crop(d["in_grid"], d["grid_dim"][0]),
            "O": crop(d["out_grid"], d["grid_dim"][-1])}


def ask(task: str, ep: dict, src: str, model: str) -> dict:
    route = "\n".join(f"{i+1}. {op}" for i, op in enumerate(ep["ops"]))
    body = QUESTION.format(src=src, I="\n".join(ep["I"]), O="\n".join(ep["O"]),
                           route=route)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        r = subprocess.run(
            [shutil.which("claude") or "claude", "-p", "--model", model,
             "--output-format", "text", "--dangerously-skip-permissions"],
            stdin=open(path), capture_output=True, text=True, timeout=300)
        out = (r.stdout or "").strip()
    except Exception as e:
        return {"error": type(e).__name__}
    finally:
        os.unlink(path)
    try:
        return json.loads(out[out.index("{"):out.rindex("}") + 1])
    except Exception:
        return {"error": "unparsed", "raw": out[:160]}


def evidence(ep: dict, verdict: dict) -> str:
    which = str(verdict.get("which", "")).strip()
    named = ""
    for tok in which.replace(",", " ").split():
        if tok.isdigit() and 1 <= int(tok) <= len(ep["ops"]):
            named += f"\n  operation {tok}: {ep['ops'][int(tok) - 1]}"
    return (f"Your route reaches the target, and that is not what is wrong with it. "
            f"One of its operations is there to put the coordinates somewhere "
            f"convenient rather than to carry out the rule:{named or ' ' + which}\n\n"
            f"{verdict.get('why', '')}\n\n"
            f"The whole route was: {' '.join(ep['ops'])}\n\n"
            f"Re-derive so that every operation is part of what the task does. Where "
            f"a selection is awkward to express in the grid's own coordinates, name "
            f"the cells directly rather than turning the grid to reach them.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a rollout directory, e.g. <draw>/whole")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--model", default="claude-fable-5")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--rearc_root", default=str(SOLAR_ROOT / "re-arc"))
    ap.add_argument("--out", default="probe_device.json")
    args = ap.parse_args()

    tasks = args.tasks or sorted({d.split("test.")[1].split(".")[0]
                                  for d in glob.glob(f"{args.root}/test.*")})

    def one(task):
        ep = episode(args.root, task)
        src = verifier_source(task, args.rearc_root)
        if ep is None or src is None:
            return None
        v = ask(task, ep, src, args.model)
        rec = {"task_id": task, "verdict": "PASS", "findings": []}
        if v.get("device"):
            rec["verdict"] = "REVISE"
            rec["findings"].append({"code": "ROUTE_USES_COORDINATE_DEVICE",
                                    "severity": "medium",
                                    "evidence": evidence(ep, v)})
        rec["_why"] = v.get("why", v.get("error", ""))[:160]
        rec["_which"] = str(v.get("which", ""))[:40]
        return rec

    recs, flagged = [], 0
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for rec in ex.map(one, tasks):
            if rec is None:
                continue
            recs.append(rec)
            if rec["findings"]:
                flagged += 1
                print(f"{rec['task_id']}  op {rec['_which']}  {rec['_why'][:70]}", flush=True)
    Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=1))
    print(f"\ntasks read {len(recs)} | coordinate device {flagged}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
