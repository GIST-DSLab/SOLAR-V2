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

Those six are all about the maker: given the answer, does it reach it honestly.
Three more are about the episode, and nothing above can see them, because the
maker is handed I and O and the reader is not. The reader gets three worked
pairs and a test input, and an episode that does not pin the rule is
unanswerable however clean its trajectory is:

    split    == 0   the colour roles are the same in all four pairs
    constant == 0   the three demonstrations do not all show the same answer
    cover    == 0   the test's answer is one the demonstrations have shown

`constant` is d9fac9be, where every episode gives the same one-cell answer four
times over: copying a demonstration is always right, so nothing in the episode
has to be read. `cover` is 27a28665, where the demonstrations answer 3, 3 and 6
and the test wants 1 -- a colour never attached to a shape. Both were found by
reading the released episodes, not by any check that existed.

`cover` only means something where the answer is an arbitrary label. 27a28665
answers a shape made of 6s and 9s with the colour 1: nothing in the input says
1, so the pairing of shape to colour exists only in the demonstrations, and a
label they never showed cannot be worked out. Where the answer is instead
assembled from colours the input already holds -- de1cd16c picks a colour out
of the grid, 1190e5a7 returns a block of one -- a reader who has the selection
rule produces an answer no demonstration used, and that is the rule working,
not a gap. So the axis asks two things first: that the answers are few and
small, and that they are labels rather than selections. Otherwise it is None. `zero` is None
wherever no target in the draw holds a 0, and `spare` is None wherever the
route was too long to ablate. An axis that could not be measured fails nobody;
it is reported as unmeasured so the count is honest.

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

# `cover` is only asked of tasks that answer with one of a few small grids.
# Above these two the answers are individual pictures rather than a vocabulary,
# and "the test's answer was never demonstrated" stops being a defect.
SMALL_CELLS = 9        # a 3x3 answer or smaller
SMALL_VOCAB = 12       # distinct answers across the task's whole draw
# ...and the answers have to be labels. A pair whose answer holds a colour its
# own input does not is one the reader could not have assembled from what is
# in front of them; where most pairs are like that, the mapping lives in the
# demonstrations alone. Under this the answers are selections out of the input
# and an unfamiliar one is derivable.
LABEL_SHARE = 0.5

EPISODE_AXES = ("split", "constant", "cover")

# axis -> (test, one-line name for the report)
BARS = {
    "solve": (lambda v: v is not None and v >= 1.0 - 1e-9, "생성된 에피소드를 다 못 맞춤"),
    "copy":  (lambda v: v is None or v <= 1e-9, "정답 그리드를 그대로 씀"),
    "idle":  (lambda v: v is None or v <= 1e-9, "아무것도 안 바꾸는 동작이 있음"),
    "dep":   (lambda v: v is None or v <= 1e-9, "정답을 흐트러뜨리면 경로가 바뀜"),
    "spare": (lambda v: v is None or v < SPARE_MIN, "빼도 정답에 닿는 동작이 있음"),
    "zero":  (lambda v: v is None or v >= 1.0 - 1e-9, "색 0을 가진 칸을 잃음"),
    "split": (lambda v: v is None or v <= 1e-9, "쌍마다 색 역할이 갈림"),
    "constant": (lambda v: v is None or v <= 1e-9,
                 "예시 세 답이 서로 같아 무엇이 답을 가르는지 안 보임"),
    "cover": (lambda v: v is None or v <= 1e-9, "test의 답을 예시가 보여준 적 없음"),
}


PAD = 10               # the value every grid is padded to 30x30 with


def _grid(flat, dim) -> tuple:
    """One grid out of a published episode, unpadded.

    The output's extent is the LAST step's grid_dim, not the input's: a task
    that resizes would otherwise be read at the input's shape and the padding
    counted as a colour. That mistake turned one real finding into a hundred
    and six the first time these episodes were measured.
    """
    h, w = dim
    return tuple(tuple(c for c in row[:w] if c != PAD) for row in flat[:h])


def _cells(g) -> int:
    return sum(len(r) for r in g)


def _colours(g) -> set:
    return {c for row in g for c in row}


_ROLES: dict[str, int | None] = {}


def declared_roles(makers_dir: Path, tid: str) -> int | None:
    """How many of the colours the hold recorded are roles.

    The hold logs every colour the generator drew, and for some makers that is
    a stream: 31aa019c's entry runs to hundreds of per-cell draws, and
    comparing the whole of it between pairs reports a disagreement on every
    episode while the one colour that matters -- the marker it paints the
    border with -- is identical in all four. Only the opening draws are role
    assignments, and sample_colors() is what says how many, so that is what is
    asked rather than anything this file could work out from the log's shape.
    """
    if tid in _ROLES:
        return _ROLES[tid]
    _ROLES[tid] = None
    path = makers_dir / tid / "grid_maker.py"
    if path.is_file():
        sys.path.insert(0, str(makers_dir))
        try:
            import importlib.util
            from holds import _maker_palette
            spec = importlib.util.spec_from_file_location(f"pf_{tid}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            got = _maker_palette(mod)
            _ROLES[tid] = got[1] if got else None
        except Exception:
            pass
        finally:
            sys.path.pop(0)
    return _ROLES[tid]


def episode_axes(episodes_root: Path, tid: str,
                 makers_dir: Path | None = None) -> dict:
    """What the published episodes of one task look like to a reader.

    Read off the episode files rather than recomputed: these are the four
    pairs someone downloading the dataset is shown, and the question is
    whether those four pin the rule.
    """
    base = str(episodes_root).rstrip("/")
    if base.endswith("/whole"):
        base = base[: -len("/whole")]
    hits = sorted(Path(base).glob(f"whole/test.{tid}.*"))
    if not hits:
        return {"split": None, "constant": None, "cover": None, "episodes": 0}
    roles = declared_roles(makers_dir, tid) if makers_dir else None
    eps, splits, labels = [], [], []
    for f in sorted(hits[0].glob("*.json"), key=lambda q: int(q.stem)):
        e = json.loads(f.read_text())
        exi = [_grid(g, d) for g, d in zip(e["ex_in"], e["ex_in_grid_dim"])]
        exo = [_grid(g, d) for g, d in zip(e["ex_out"], e["ex_out_grid_dim"])]
        ti = _grid(e["in_grid"], e["grid_dim"][0])
        to = _grid(e["out_grid"], e["grid_dim"][-1])
        eps.append((exo, to))
        labels += [not _colours(b) <= _colours(a)
                   for a, b in list(zip(exi, exo)) + [(ti, to)]]
        hold = (e.get("desc") or {}).get("palette_hold")
        if hold and roles:
            # Compare the roles the maker declared, and no further: past them
            # the entries are per-instance object colours whose count follows
            # the picture, so a pair that drew fewer would read as a
            # disagreement on length alone, and a maker that draws per cell
            # would read as one on every episode.
            k = min([roles] + [len(q) for q in hold])
            splits.append(k == 0 or len({tuple(q[:k]) for q in hold}) > 1)
    if not eps:
        return {"split": None, "constant": None, "cover": None, "episodes": 0}
    n = len(eps)
    out = {
        "episodes": n,
        "split": (sum(splits) / len(splits)) if splits else None,
        "held": len(splits),
        "constant": sum(1 for exo, _ in eps if len(set(exo)) == 1) / n,
        "cover": None,
    }
    answers = [o for exo, to in eps for o in exo] + [to for _, to in eps]
    if max(_cells(o) for o in answers) <= SMALL_CELLS \
            and len(set(answers)) <= SMALL_VOCAB \
            and sum(labels) / len(labels) > LABEL_SHARE:
        out["cover"] = sum(1 for exo, to in eps if to not in set(exo)) / n
    return out


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
            v = (row.get("episode") or {}).get(axis, s.get(axis)) \
                if axis in EPISODE_AXES else s.get(axis)
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
    for row in rows:
        row["episode"] = episode_axes(episodes_root, row["task_id"], makers_dir)
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
    by_axis: dict[str, int] = {}
    for u in unmeasured:
        by_axis[u["axis"]] = by_axis.get(u["axis"], 0) + 1
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
        "unmeasured_by_axis": by_axis,
        # Not a fault, and not silence either. A task whose maker declares no
        # colour roles is indistinguishable here from one whose roles the hold
        # failed to capture, and the difference matters, so the count is
        # carried rather than folded into the unmeasured total.
        "no_hold": sum(1 for r in rows
                       if (r.get("episode") or {}).get("split") is None
                       and (r.get("episode") or {}).get("episodes")),
        "draw": (draw.parent if draw.name == "whole" else draw).name,
        "bar": {"solve": 1.0, "copy": 0.0, "idle": 0.0, "dep": 0.0,
                "spare": f"< {SPARE_MIN}", "zero": 1.0,
                "split": 0.0, "constant": 0.0, "cover": 0.0},
    }


def render(summary: dict) -> str:
    """The verdict, then the faults, then what could not be looked at.

    The unmeasured count is broken out per axis rather than given as one
    number: `cover` is None on nearly every task by design -- it only means
    something where the answers are a small vocabulary -- and a single total
    of four hundred makes that read as four hundred blind spots.
    """
    lines = [f"  {summary['passed']}/{summary['tasks']} 통과"]
    for f in summary["findings"]:
        v = "" if f["value"] is None else f" {f['value']:.2f}"
        lines.append(f"  {f['task_id']}  {f['axis']}{v}  — {f['why']}")
    un = summary.get("unmeasured_by_axis") or {}
    if un:
        lines.append("  측정 불가: " + ", ".join(
            f"{a} {n}" for a, n in sorted(un.items(), key=lambda kv: -kv[1])))
    if summary.get("no_hold"):
        lines.append(f"  색 유지 장치가 안 걸린 과제 {summary['no_hold']} "
                     f"— 역할을 선언하지 않은 과제인지 놓친 것인지 여기서는 못 가림")
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
