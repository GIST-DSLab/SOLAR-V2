"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


# ----------------------------------------------------------------------------
# colors: the generator only samples `bgc` at random (1/2/3/7 are hardcoded roles)
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 7)])
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _sign(x):
    return (x > 0) - (x < 0)


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    max_h = max(8, min(30, int(max_h)))
    max_w = max(8, min(30, int(max_w)))

    for _attempt in range(64):
        h = _unifint(diff_lb, diff_ub, (8, max_h))
        w = _unifint(diff_lb, diff_ub, (8, max_w))

        go = [[bgc] * w for _ in range(h)]

        rdi = random.randint(1, h - 2)
        rdj = random.randint(1, w - 2)
        rd = (rdi, rdj)

        # keep the two centers far enough apart that their neighbourhoods are
        # disjoint -> each satellite has an unambiguous owner / destination
        cands = [
            (i, j)
            for i in range(1, h - 1)
            for j in range(1, w - 1)
            if max(abs(i - rdi), abs(j - rdj)) >= 3
        ]
        if not cands:
            continue
        bd = random.choice(cands)
        bdi, bdj = bd

        nbrs_rd = [
            (rdi + di, rdj + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        ]
        nbrs_bd = [
            (bdi + di, bdj + dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        ]

        go[rdi][rdj] = 2
        go[bdi][bdj] = 1

        ngd = _unifint(diff_lb, diff_ub, (1, 8))
        nod = _unifint(diff_lb, diff_ub, (1, 8))
        gd = random.sample(nbrs_rd, ngd)
        od = random.sample(nbrs_bd, nod)

        for (i, j) in gd:
            go[i][j] = 3
        for (i, j) in od:
            go[i][j] = 7

        mpr = {}
        for (i, j) in gd:
            mpr[(i, j)] = (3, (_sign(i - rdi), _sign(j - rdj)))
        for (i, j) in od:
            mpr[(i, j)] = (7, (_sign(i - bdi), _sign(j - bdj)))

        forbidden = set(nbrs_rd) | set(nbrs_bd) | {rd, bd}

        gi = [row[:] for row in go]
        ub = max(1, (len(gd) + len(od)) * ((h + w) // 5))
        ndist = _unifint(diff_lb, diff_ub, (1, ub))

        moved = 0
        for _ in range(ndist):
            options = []
            for loc, (col, d) in mpr.items():
                ii, jj = loc[0] + d[0], loc[1] + d[1]
                if 0 <= ii < h and 0 <= jj < w and gi[ii][jj] == bgc \
                        and (ii, jj) not in forbidden:
                    options.append((loc, col, d))
            if not options:
                break
            loc, col, d = random.choice(options)
            del mpr[loc]
            newloc = (loc[0] + d[0], loc[1] + d[1])
            mpr[newloc] = (col, d)
            gi[loc[0]][loc[1]] = bgc
            gi[newloc[0]][newloc[1]] = col
            moved += 1

        if moved == 0:
            continue

        return {
            "input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go),
        }

    # fallback (should be unreachable in practice)
    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ----------------------------------------------------------------------------
# operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the only colour that is not one of the four fixed roles
    cnt = Counter(I.flatten().tolist())
    bgc = 0
    for col, _n in cnt.most_common():
        if col not in (1, 2, 3, 7):
            bgc = int(col)
            break

    c1 = tuple(int(v) for v in np.argwhere(I == 1)[0])   # anchor for the 7s
    c2 = tuple(int(v) for v in np.argwhere(I == 2)[0])   # anchor for the 3s

    ops, sels = [], []

    # every satellite slides back along its own ray until it touches its anchor
    for col, ctr in ((7, c1), (3, c2)):
        cells = [tuple(int(v) for v in p) for p in np.argwhere(I == col)]
        # farthest satellite first, so each object is handled start-to-finish
        cells.sort(key=lambda rc: -max(abs(rc[0] - ctr[0]), abs(rc[1] - ctr[1])))
        for (r, c) in cells:
            dr = _sign(r - ctr[0])
            dc = _sign(c - ctr[1])
            dest = (ctr[0] + dr, ctr[1] + dc)
            tr = dest[0] - r
            tc = dest[1] - c
            if tr == 0 and tc == 0:
                continue                      # already touching the anchor
            sr = _sign(tr)
            sc = _sign(tc)
            n = max(abs(tr), abs(tc))
            cur = (r, c)
            first = True
            for _k in range(n):
                if sr != 0:
                    ops.append(20 if sr < 0 else 21)
                    sels.append(sel_of([cur]) if first else sel_of([]))
                    first = False
                    cur = (cur[0] + sr, cur[1])
                if sc != 0:
                    ops.append(23 if sc < 0 else 22)
                    sels.append(sel_of([cur]) if first else sel_of([]))
                    first = False
                    cur = (cur[0], cur[1] + sc)
            # ARCLE left the object's ORIGINAL footprint at 0 (the path it glided
            # over was restored automatically); repair just that one cell
            if bgc != 0:
                ops.append(bgc)
                sels.append(sel_of([(r, c)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])   # full-grid rectangle: submit
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
