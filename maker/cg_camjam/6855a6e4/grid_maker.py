"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=5, boxc=2, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))

    def unif(a, b):
        b = max(a, b)
        return a + int(round((b - a) * random.uniform(diff_lb, diff_ub)))

    hub = max_w if transpose else max_h
    wub = max_h if transpose else max_w
    h = unif(10, min(30, hub))
    w = unif(4, min(30, wub))
    fullh = unif(10, h)
    fullw = unif(3, w)
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    canvi = [[bgc] * fullw for _ in range(fullh)]
    canvo = [[bgc] * fullw for _ in range(fullh)]
    objh = (fullh // 2 - 3) // 2
    br = {(objh + 1, j) for j in range(fullw)} | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = [(i, j) for i in range(0, objh) for j in range(1, fullw - 1)]
    for k in range(2):
        for (i, j) in br:
            canvi[i][j] = boxc
            canvo[i][j] = boxc
        tot = objh * (fullw - 2)
        ncellsd = unif(0, tot // 2)
        ncells = random.choice((ncellsd, tot - ncellsd))
        ncells = min(max(1, ncells), tot)
        cells = random.sample(cands, ncells)
        for (i, j) in cells:
            canvi[i][j] = objc
        mx = max(i for i, _ in cells)
        for (i, j) in cells:
            canvo[mx - i + objh + 3][j] = objc   # mirrored, flush against the stub row (verifier rule)
        canvi = canvi[::-1]
        canvo = canvo[::-1]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for i in range(fullh):
        for j in range(fullw):
            gi[loci + i][locj + j] = canvi[i][j]
            go[loci + i][locj + j] = canvo[i][j]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    colors = sorted(int(c) for c in np.unique(I))

    # Frame: two full parallel rails + four inward stubs, nothing else.
    boxc, horizontal = None, True
    for c in colors:
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if not all((r in (r0, r1)) or (cc in (c0, c1)) for r, cc in pts.tolist()):
            continue
        rows_full = bool((I[r0, c0:c1 + 1] == c).all() and (I[r1, c0:c1 + 1] == c).all())
        cols_full = bool((I[r0:r1 + 1, c0] == c).all() and (I[r0:r1 + 1, c1] == c).all())
        stubs_h = hh >= 4 and ww >= 3 and all(
            I[a, b] == c for a, b in ((r0 + 1, c0), (r0 + 1, c1), (r1 - 1, c0), (r1 - 1, c1)))
        stubs_v = ww >= 4 and hh >= 3 and all(
            I[a, b] == c for a, b in ((r0, c0 + 1), (r1, c0 + 1), (r0, c1 - 1), (r1, c1 - 1)))
        if rows_full and stubs_h and len(pts) == 2 * ww + 4:
            boxc, horizontal = c, True
            break
        if cols_full and stubs_v and len(pts) == 2 * hh + 4:
            boxc, horizontal = c, False
            break
    bx = np.argwhere(I == boxc)
    br0, br1 = int(bx[:, 0].min()), int(bx[:, 0].max())
    bc0, bc1 = int(bx[:, 1].min()), int(bx[:, 1].max())
    # Background: what fills the frame's interior centre.
    bgc = int(I[(br0 + br1) // 2, (bc0 + bc1) // 2])
    objc = [c for c in colors if c not in (boxc, bgc)][0]
    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]

    ops, sels = [], []
    # Reading: each outside pattern is REFLECTED ACROSS ITS OWN RAIL (one unit:
    # pattern + gap + rail + mirror zone). The slab symmetric about the rail line,
    # restricted to the pattern's span along the rail, is flipped whole: the rail
    # maps onto itself and the pattern lands mirrored inside the frame.
    if horizontal:
        sides = [([p for p in obj if p[0] < br0], br0), ([p for p in obj if p[0] > br1], br1)]
    else:
        sides = [([p for p in obj if p[1] < bc0], bc0), ([p for p in obj if p[1] > bc1], bc1)]
    for cells, rail in sides:
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        ax = rs if horizontal else cs
        outward = -1 if rail == (br0 if horizontal else bc0) else 1
        far = min(ax) if outward < 0 else max(ax)
        near = max(ax) if outward < 0 else min(ax)
        a, b = sorted((far, 2 * rail - far))
        # 1. reflect across the rail: flip the full slab rectangle (background included)
        if horizontal:
            ops.append(27); sels.append([a, min(cs), b - a, max(cs) - min(cs)])
            moved = [(2 * rail - r, c) for r, c in cells]
        else:
            ops.append(26); sels.append([min(rs), a, max(rs) - min(rs), b - a])
            moved = [(r, 2 * rail - c) for r, c in cells]
        # 2. if the pattern did not touch the spacer line, slide the reflected
        #    pattern back toward its rail until flush against the stub line.
        steps = abs(rail - near) - 2
        if steps > 0 and objc == 0:
            # ARCLE's Move treats 0 as empty, so a colour-0 pattern cannot be carried:
            # lift its reflected cells to background and lay it down flush.
            if horizontal:
                dest = [(r + outward * steps, c) for r, c in moved]
            else:
                dest = [(r, c + outward * steps) for r, c in moved]
            ops.append(bgc); sels.append(sel_of(sorted(moved)))
            ops.append(0); sels.append(sel_of(sorted(dest)))
        elif steps > 0:
            if horizontal:
                op = 20 if outward < 0 else 21
                d = (-1, 0) if outward < 0 else (1, 0)
            else:
                op = 23 if outward < 0 else 22
                d = (0, -1) if outward < 0 else (0, 1)
            src = list(moved)
            cur = list(moved)
            for k in range(steps):
                ops.append(op); sels.append(sel_of(cur if k == 0 else []))   # grab once, then empties
                cur = [(r + d[0], c + d[1]) for r, c in cur]
            hole = sorted(set(src) - set(cur))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
