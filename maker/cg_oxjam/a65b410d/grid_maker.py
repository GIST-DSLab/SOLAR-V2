"""
ARC Task: a65b410d (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np


def sample_colors(num_examples=None) -> dict:
    # Output colours 1 and 3 are hardcoded by the task, so the background and
    # line colours are sampled from the other eight colours.
    colopts = [c for c in range(10) if c not in (1, 3)]
    bgc = random.choice(colopts)
    linc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    colopts = [c for c in range(10) if c not in (1, 3)]
    if bgc is None:
        bgc = random.choice(colopts)
    if linc is None or linc == bgc:
        linc = random.choice([c for c in colopts if c != bgc])

    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    mpi, mpj = h // 2, w // 2
    devi = unifint(diff_lb, diff_ub, (0, mpi))
    devj = unifint(diff_lb, diff_ub, (0, mpj))
    if random.choice((True, False)):
        loci, locj = devi, devj
    else:
        loci, locj = h - devi, w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)

    gi = np.full((h, w), bgc, dtype=int)
    gi[loci, 0:locj + 1] = linc
    go = gi.copy()

    # Blue (1) wedge: each row below the line gets a run from column 0 that is
    # one cell shorter than the row above it.
    k = 1
    while loci + k < h:
        e = locj - k
        if e < 0:
            break
        go[loci + k, 0:e + 1] = 1
        k += 1

    # Green (3) wedge: each row above the line gets a run from column 0 that is
    # one cell longer than the row below it, clipped at the right edge.
    k = 1
    while loci - k >= 0:
        e = min(locj + k, w - 1)
        go[loci - k, 0:e + 1] = 3
        k += 1

    rk = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -rk)  # rot90 in the DSL is clockwise
    go = np.rot90(go, -rk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # How this compares with the code-golf reading: both read the task as one
    # staircase. Every row is a run that starts at the edge where the line is
    # anchored, and its length changes by exactly one from row to row. Above
    # the tip the runs are coloured 3, the line row stays as it is, and below
    # the tip the runs are coloured 1. Neither reading changes what counts as
    # a unit, the order of the work, or which quantity is held fixed.
    # So this function keeps the existing per-row-run route: it paints each
    # run as a full rectangle, starting next to the line and working outward.
    # Rows whose runs are clipped to the full grid width merge into one
    # rectangle. Every selection comes from I alone (the line's position and
    # its tip). O is used only for the Submit size, which equals I's size.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # The line colour is the rarest colour in I.
    cnt = Counter(I.flatten().tolist())
    linc = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]

    # Normalise the orientation: find the rotation in which the line is a row
    # anchored at column 0.
    idx = np.arange(h * w).reshape(h, w)
    J, Jidx = I, idx
    for k in range(4):
        Jc = np.rot90(I, k)
        cells = np.argwhere(Jc == linc)
        if len(set(cells[:, 0].tolist())) == 1 and cells[:, 1].min() == 0:
            J, Jidx = Jc, np.rot90(idx, k)
            break
    H, W = J.shape
    cells = np.argwhere(J == linc)
    Rl = int(cells[0, 0])
    Cmax = int(cells[:, 1].max())  # free tip of the line

    def rows_of(sign):
        # sign -1 lists the rows above the line, sign +1 the rows below it.
        seq = []
        k = 1
        while 0 <= Rl + sign * k <= H - 1:
            if sign < 0:
                e = min(Cmax + k, W - 1)
            else:
                e = Cmax - k
                if e < 0:
                    break
            seq.append((Rl + sign * k, e))
            k += 1
        return seq

    def group(seq):
        # Merge consecutive rows that end at the same column.
        out = []
        for r, e in seq:
            if out and out[-1][1] == e:
                out[-1][0].append(r)
            else:
                out.append(([r], e))
        return out

    ops, sels = [], []

    def paint(seq, color):
        for rows, e in group(seq):
            a, b = min(rows), max(rows)
            pts = []
            for R in (a, b):
                for C in (0, e):
                    f = int(Jidx[R, C])
                    pts.append((f // w, f % w))
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            # The intended cells fill this whole rectangle, so a bbox is valid.
            ops.append(color)
            sels.append([min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)])

    paint(rows_of(-1), 3)  # wedge on the side the tip leans toward: growing runs
    paint(rows_of(+1), 1)  # opposite side: shrinking runs

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a65b410d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a65b410d"
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
                                f"for task a65b410d"
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
                    f"Failed to build a complete episode for task a65b410d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a65b410d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
