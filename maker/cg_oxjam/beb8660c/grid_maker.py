"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

ROTS = ["identity", "rot90", "rot180", "rot270"]


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    variants = [{"rot": r} for r in ROTS]
    if n_ex >= len(variants):
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(variants, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, rot=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if rot is None:
        rot = random.choice(ROTS)
    swap = rot in ("rot90", "rot270")
    H_lim = max_w if swap else max_h
    W_lim = max_h if swap else max_w
    w = _unifint(diff_lb, diff_ub, (3, min(W_lim, H_lim)))
    h = _unifint(diff_lb, diff_ub, (w, H_lim))
    # line colours: never bgc, never 8, never 0 (a 0-coloured object cannot be moved in ARCLE)
    remcols = [c for c in cols if c != bgc and c != 0]
    kmax = min(8, w - 1, h - 2, len(remcols))  # h-2: keep at least one free lane for parking
    k = _unifint(diff_lb, diff_ub, (1, kmax))
    co = random.sample(remcols, k)
    wds = sorted(random.sample(range(1, w), k))
    gi = np.full((h, w), bgc, dtype=int)
    for j, (c, l) in enumerate(zip(co, wds)):
        rr = h - k - 1 + j
        gi[rr, 0:l] = c
    gi[h - 1, :] = 8
    go = gi[:, ::-1].copy()
    rows = [list(r) for r in gi[:-1]]
    random.shuffle(rows)
    gif = []
    for r in rows + [list(go[-1])]:
        nbc = r.count(bgc)
        ofs = random.randint(0, nbc)
        gif.append(r[-ofs:] + r[:-ofs] if ofs else r[:])
    gi = np.array(gif, dtype=int)[:, ::-1]
    k_rot = {"identity": 0, "rot90": 3, "rot180": 2, "rot270": 1}[rot]  # DSL rot90 = clockwise
    gi = np.rot90(gi, k_rot)
    go = np.rot90(go, k_rot)
    return {"input": [list(map(int, r)) for r in gi], "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (as in sorted(map(sorted, g))): every line is a rigid object.
    # Phase 1 = "sort within each lane": each line slides along its own lane to the aligned end.
    # Phase 2 = "sort the lanes": lines are translated across lanes, longest next to the 8 wall.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- the 8 wall: the full edge row/column of 8 ---
    if np.all(I[hi - 1, :] == 8):
        edge = "bottom"
    elif np.all(I[0, :] == 8):
        edge = "top"
    elif np.all(I[:, 0] == 8):
        edge = "left"
    else:
        edge = "right"
    horiz = edge in ("top", "bottom")
    N = wi if horiz else hi          # length along a line
    M = (hi if horiz else wi) - 1    # number of lanes parallel to the wall (excluding it)

    def cell(d, a):                  # d = distance from the wall (1 = adjacent), a = along-coordinate
        if edge == "bottom":
            return (hi - 1 - d, a)
        if edge == "top":
            return (d, a)
        if edge == "left":
            return (a, d)
        return (a, wi - 1 - d)

    def lane(d):
        return [I[cell(d, a)] for a in range(N)]

    # --- background: the non-8 colour present in every lane (every line is shorter than its lane) ---
    common = None
    for d in range(1, M + 1):
        s = set(lane(d)) - {8}
        common = s if common is None else common & s
    if len(common) == 1:
        bgc = next(iter(common))
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- the lines: one segment per lane ---
    lines = []
    for d in range(1, M + 1):
        ln = lane(d)
        idx = [a for a in range(N) if ln[a] != bgc]
        if idx:
            lines.append({"color": ln[idx[0]], "d": d, "a": idx[0], "L": len(idx)})

    align_end = edge in ("bottom", "left")   # True -> segment ends at a = N-1, else starts at a = 0
    along_pos = 22 if horiz else 21          # step a+1
    along_neg = 23 if horiz else 20          # step a-1
    perp_out = {"bottom": 20, "top": 21, "left": 22, "right": 23}[edge]   # step d+1
    perp_in = {"bottom": 21, "top": 20, "left": 23, "right": 22}[edge]    # step d-1

    def cells_of(ln):
        return [cell(ln["d"], a) for a in range(ln["a"], ln["a"] + ln["L"])]

    def slide(ln, steps):
        """steps: list of (op, dd, da). One grab, then empty selections."""
        if not steps:
            return
        src = cells_of(ln)
        if ln["color"] == 0:  # ARCLE cannot carry a 0-coloured object: erase and repaint
            for _, dd, da in steps:
                ln["d"] += dd
                ln["a"] += da
            dst = cells_of(ln)
            hole = sorted(set(src) - set(dst))
            if hole:
                ops.append(int(bgc)); sels.append(sel_of(hole))
            ops.append(0); sels.append(sel_of(dst))
            return
        for i, (op, dd, da) in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(cells_of(ln)) if i == 0 else sel_of([]))
            ln["d"] += dd
            ln["a"] += da
        hole = sorted(set(src) - set(cells_of(ln)))
        if bgc != 0 and hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))

    # --- phase 1: within its own lane, each line slides to the aligned end ---
    for ln in lines:
        target_a = N - ln["L"] if align_end else 0
        delta = target_a - ln["a"]
        op = along_pos if delta > 0 else along_neg
        sg = 1 if delta > 0 else -1
        slide(ln, [(op, 0, sg)] * abs(delta))

    # --- phase 2: lanes are reordered, longest next to the wall ---
    order = sorted(lines, key=lambda x: -x["L"])
    for i, ln in enumerate(order):
        ln["t"] = i + 1

    def perp_move(ln, to_d):
        delta = to_d - ln["d"]
        op = perp_out if delta > 0 else perp_in
        sg = 1 if delta > 0 else -1
        slide(ln, [(op, sg, 0)] * abs(delta))

    while True:
        pending = [ln for ln in order if ln["d"] != ln["t"]]
        if not pending:
            break
        occ = {ln["d"] for ln in lines if ln["d"] is not None}
        ready = [ln for ln in pending if ln["t"] not in occ]
        if ready:
            ln = ready[0]
            if ln["d"] is None:  # a lifted line is set down in its lane
                ln["d"] = ln["t"]
                ops.append(int(ln["color"])); sels.append(sel_of(cells_of(ln)))
            else:
                perp_move(ln, ln["t"])
            continue
        # every pending target is held by another pending line (a cycle): park one in a free lane
        ln = pending[0]
        free = [d for d in range(1, M + 1) if d not in occ]
        if free:
            perp_move(ln, min(free, key=lambda d: (abs(d - ln["d"]), d)))
        else:  # no free lane (excluded by generate()): lift the line off the board
            ops.append(int(bgc)); sels.append(sel_of(cells_of(ln)))
            ln["d"] = None

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task beb8660c"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
