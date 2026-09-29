#!/usr/bin/env python3
"""Run the maker gate over every task in a draw before it is published.

The gate in choose_maker.py answers "is this candidate better than the
incumbent", and until now it was only ever asked about the tasks a promotion
was being decided for. Nine makers that consulted the answer, and thirteen that
emitted an operation their route reaches the answer without, sat in the release
for months because nobody proposed a replacement for them, so nobody measured
them. The first sweep of all four hundred found thirty-two faults.

This module is that sweep, made a condition of export rather than something to
remember. It scores the maker that is actually in the tree against the episodes
that are actually going to be published, and reports every task that fails the
release bar -- which is the promotion bar with no incumbent to be better than:

    solve == 1      answers every episode the draw kept
    copy  == 0      does not carry its parameters off the target grid
    idle  == 0      every operation it emits changes something
    dep   == 0      the route does not change when the answer is disturbed
    spare <  0.5    no operation the route reaches the answer without
    zero  == 1      loses no cell to ARCLE reading colour 0 as nothing there

`zero` is None wherever no target in the draw holds a 0, and `spare` is None
wherever the route was too long to ablate. An axis that could not be measured
fails nobody; it is reported as unmeasured so the count is honest.

    python pipeline/preflight.py --episodes_root $SOLAR_DATA_ROOT/ARC_rearc_draw19 \
        --makers maker/arc-agi-1 --report /path/to/preflight.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SOLAR_ROOT = Path(__file__).parent.parent.resolve()

# The release bar. `spare` is a rate over sampled instances and 0.5 is the same
# threshold choose_maker promotes against: below a majority the removable
# operation is the draw's doing -- a colour laid on cells that one instance
# already had -- and above it the route emits the operation whatever it is
# given, which is the maker's.
SPARE_MIN = 0.5

# axis -> (test, one-line name for the report)
BARS = {
    "solve": (lambda v: v is not None and v >= 1.0 - 1e-9, "생성된 에피소드를 다 못 맞춤"),
    "copy":  (lambda v: v is None or v <= 1e-9, "정답 그리드를 그대로 씀"),
    "idle":  (lambda v: v is None or v <= 1e-9, "아무것도 안 바꾸는 동작이 있음"),
    "dep":   (lambda v: v is None or v <= 1e-9, "정답을 흐트러뜨리면 경로가 바뀜"),
    "spare": (lambda v: v is None or v < SPARE_MIN, "빼도 정답에 닿는 동작이 있음"),
    "zero":  (lambda v: v is None or v >= 1.0 - 1e-9, "색 0을 가진 칸을 잃음"),
}


def task_ids(makers_dir: Path, exclude: set[str] | None = None) -> list[str]:
    """Every maker in the directory, in the order the gate will see them."""
    skip = exclude or set()
    return sorted(p.name for p in makers_dir.iterdir()
                  if p.is_dir() and p.name != "__pycache__" and p.name not in skip)


def sweep(episodes_root: Path, makers_dir: Path, label: str,
          report: Path, exclude: set[str] | None = None,
          python: str = sys.executable) -> list[dict]:
    """Score every maker in `makers_dir` against `episodes_root`.

    Runs choose_maker with the maker set as its own only candidate: with nothing
    to compare against, the winner is uninteresting and the scores are the whole
    point.
    """
    tasks = task_ids(makers_dir, exclude)
    if not tasks:
        raise SystemExit(f"no makers under {makers_dir}")
    report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(SOLAR_ROOT / "pipeline" / "choose_maker.py"),
           "--candidates", label, "--tasks", *tasks,
           "--episodes_root", str(episodes_root), "--out", str(report)]
    proc = subprocess.run(cmd, cwd=SOLAR_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"the gate exited {proc.returncode}; nothing was exported")
    return json.loads(report.read_text())


def faults(rows: list[dict], label: str) -> tuple[list[dict], list[dict]]:
    """Split the sweep into (tasks that fail the bar, axes nobody could measure).

    A maker that did not load at all is the loudest possible failure and is
    reported as one, not skipped for having no scores to check.
    """
    bad, unmeasured = [], []
    for row in rows:
        s = (row.get("scores") or {}).get(label)
        tid = row["task_id"]
        if not s or not s.get("loaded"):
            bad.append({"task_id": tid, "axis": "load",
                        "value": None, "why": "maker가 로드되지 않음"})
            continue
        for axis, (ok, why) in BARS.items():
            v = s.get(axis)
            if v is None:
                unmeasured.append({"task_id": tid, "axis": axis})
            elif not ok(v):
                bad.append({"task_id": tid, "axis": axis,
                            "value": round(float(v), 3), "why": why})
    return bad, unmeasured


def check(episodes_root: Path, makers_dir: Path, label: str, report: Path,
          exclude: set[str] | None = None, reuse: bool = False) -> dict:
    """Sweep and judge. Returns a summary fit to store in the release manifest.

    `reuse` takes an existing report as-is. It is for re-running an export whose
    sweep already passed, never for skipping a sweep that has not been run: the
    caller is responsible for knowing the report is newer than the makers.
    """
    rows = (json.loads(report.read_text()) if reuse and report.is_file()
            else sweep(episodes_root, makers_dir, label, report, exclude))
    if not rows:
        # A sweep that scored nothing is not a sweep that found nothing. The
        # gate skips a task it cannot draw instances for, and if it skips all
        # of them the summary reads "0/0 passed", which is how an unchecked
        # release would come to carry a clean bill.
        raise SystemExit(
            f"the gate scored no tasks under {makers_dir}; it draws instances "
            f"from each task's RE-ARC generator, so a maker set whose names "
            f"are not RE-ARC task ids cannot be swept")
    bad, unmeasured = faults(rows, label)
    # The draw's folder name, not its path: this summary is written into
    # release_manifest.json, which is published, and nobody downloading the
    # dataset needs the directory layout of the machine that built it.
    draw = Path(str(episodes_root).rstrip("/"))
    return {
        "tasks": len(rows),
        "passed": len(rows) - len({f["task_id"] for f in bad}),
        "failed": sorted({f["task_id"] for f in bad}),
        "findings": bad,
        "unmeasured": len(unmeasured),
        "draw": (draw.parent if draw.name == "whole" else draw).name,
        "bar": {"solve": 1.0, "copy": 0.0, "idle": 0.0, "dep": 0.0,
                "spare": f"< {SPARE_MIN}", "zero": 1.0},
    }


def render(summary: dict) -> str:
    lines = [f"  {summary['passed']}/{summary['tasks']} 통과"
             f" (측정 불가 축 {summary['unmeasured']}개)"]
    for f in summary["findings"]:
        v = "" if f["value"] is None else f" {f['value']:.2f}"
        lines.append(f"  {f['task_id']}  {f['axis']}{v}  — {f['why']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_root", required=True)
    ap.add_argument("--makers", default="maker/arc-agi-1")
    ap.add_argument("--label", default=None,
                    help="what the maker set is called to choose_maker "
                         "(default: the makers directory's own name)")
    ap.add_argument("--report", required=True)
    ap.add_argument("--reuse", action="store_true",
                    help="judge an existing report instead of sweeping again")
    args = ap.parse_args()

    makers = Path(args.makers)
    if not makers.is_absolute():
        makers = SOLAR_ROOT / makers
    summary = check(Path(args.episodes_root).expanduser().resolve(), makers,
                    args.label or makers.name, Path(args.report).expanduser(),
                    reuse=args.reuse)
    print(render(summary))
    sys.exit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
