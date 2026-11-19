"""
ARC Task: d89b689b (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    bgc, sqc, a, b, c, d = random.sample(cols, 6)
    return {"bgc": bgc, "sqc": sqc, "a": a, "b": b, "c": c, "d": d}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None,
             a=None, b=None, c=None, d=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        lo, hi = rng
        if hi <= lo:
            return lo
        x = random.uniform(lb, ub)
        return int(round(lo + x * (hi - lo)))

    if None in (bgc, sqc, a, b, c, d):
        cs = sample_colors()
        bgc, sqc, a, b, c, d = cs["bgc"], cs["sqc"], cs["a"], cs["b"], cs["c"], cs["d"]

    h = unifint(diff_lb, diff_ub, (4, max(4, max_h)))
    w = unifint(diff_lb, diff_ub, (4, max(4, max_w)))
    loci = random.randint(1, h - 3)
    locj = random.randint(1, w - 3)

    go = [[bgc] * w for _ in range(h)]
    go[loci][locj] = a
    go[loci][locj + 1] = b
    go[loci + 1][locj] = c
    go[loci + 1][locj + 1] = d

    inds = [(i, j) for i in range(h) for j in range(w)]

    def man(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    aopts = sorted([ij for ij in inds if ij[0] < loci and ij[1] < locj],
                   key=lambda ij: man(ij, (loci, locj)))
    bopts = sorted([ij for ij in inds if ij[0] < loci and ij[1] > locj + 1],
                   key=lambda ij: man(ij, (loci, locj + 1)))
    copts = sorted([ij for ij in inds if ij[0] > loci + 1 and ij[1] < locj],
                   key=lambda ij: man(ij, (loci + 1, locj)))
    dopts = sorted([ij for ij in inds if ij[0] > loci + 1 and ij[1] > locj + 1],
                   key=lambda ij: man(ij, (loci + 1, locj + 1)))

    loca = aopts[unifint(diff_lb, diff_ub, (0, len(aopts) - 1))]
    locb = bopts[unifint(diff_lb, diff_ub, (0, len(bopts) - 1))]
    locc = copts[unifint(diff_lb, diff_ub, (0, len(copts) - 1))]
    locd = dopts[unifint(diff_lb, diff_ub, (0, len(dopts) - 1))]

    gi = [[bgc] * w for _ in range(h)]
    for i in (loci, loci + 1):
        for j in (locj, locj + 1):
            gi[i][j] = sqc
    gi[loca[0]][loca[1]] = a
    gi[locb[0]][locb[1]] = b
    gi[locc[0]][locc[1]] = c
    gi[locd[0]][locd[1]] = d
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared with the golf entry): each isolated pixel SLIDES into the
    2x2 block, landing on the block corner nearest to it, and covers that corner.
    The pixel keeps its shape and colour, so this is a translation and is done
    with Move ops: grab the pixel, then keep moving with empty selections.
    Exception: a pixel of colour 0 can't be moved, because ARCLE treats 0 as
    transparent. That pixel is stamped onto its corner and its old cell is
    cleared instead. Nothing here is read from O."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = int(np.bincount(I.flatten()).argmax())

    # the 2x2 uniform non-background block
    sq_r = sq_c = None
    for r in range(hi - 1):
        for c in range(wi - 1):
            v = int(I[r, c])
            if v != bgc and I[r, c + 1] == v and I[r + 1, c] == v and I[r + 1, c + 1] == v:
                sq_r, sq_c = r, c
                break
        if sq_r is not None:
            break

    corners = [(sq_r, sq_c), (sq_r, sq_c + 1), (sq_r + 1, sq_c), (sq_r + 1, sq_c + 1)]
    block = set(corners)

    sats = [(r, c, int(I[r, c]))
            for r in range(hi) for c in range(wi)
            if int(I[r, c]) != bgc and (r, c) not in block]

    assigned = []
    for (r, c, col) in sats:
        k = min(range(4), key=lambda i: abs(corners[i][0] - r) + abs(corners[i][1] - c))
        assigned.append((k, r, c, col))
    assigned.sort(key=lambda t: t[0])

    for (k, r, c, col) in assigned:
        tr, tc = corners[k]
        if col == 0:
            # A 0-coloured pixel is transparent to Move: stamp it on the corner, then clear its old cell
            ops.append(0); sels.append(sel_of([(tr, tc)]))
            ops.append(bgc); sels.append(sel_of([(r, c)]))
            continue

        dr, dc = tr - r, tc - c
        steps = []
        steps += [21 if dr > 0 else 20] * abs(dr)
        steps += [22 if dc > 0 else 23] * abs(dc)

        first = True
        for op in steps:
            ops.append(op)
            if first:
                sels.append(sel_of([(r, c)]))   # grab the pixel
                first = False
            else:
                sels.append(sel_of([]))          # keep sliding the same pixel

        # Only the pixel's starting cell is left at 0; repair it if the background isn't 0
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of([(r, c)]))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d89b689b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d89b689b"
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
                                f"for task d89b689b"
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
                    f"Failed to build a complete episode for task d89b689b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d89b689b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
