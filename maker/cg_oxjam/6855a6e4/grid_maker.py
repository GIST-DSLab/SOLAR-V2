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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, boxc=None, transpose=None) -> dict:
    if bgc is None or objc is None or boxc is None:
        bgc, objc, boxc = random.sample(range(10), 3)
    if transpose is None:
        transpose = random.choice([True, False])
    hub = max_w if transpose else max_h
    wub = max_h if transpose else max_w
    hub, wub = min(hub, 30), min(wub, 30)
    if hub < 10 or wub < 4:
        raise ValueError("grid too small")
    h = random.randint(10, hub)
    w = random.randint(4, wub)
    fullh = random.randint(10, h)
    fullw = random.randint(3, w)
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
    canvi = [[bgc] * fullw for _ in range(fullh)]
    canvo = [[bgc] * fullw for _ in range(fullh)]
    objh = (fullh // 2 - 3) // 2
    br = [(objh + 1, j) for j in range(fullw)] + [(objh + 2, 0), (objh + 2, fullw - 1)]
    cands = [(i, j) for i in range(0, objh) for j in range(1, fullw - 1)]
    for _ in range(2):
        for (i, j) in br:
            canvi[i][j] = boxc
            canvo[i][j] = boxc
        tot = objh * (fullw - 2)
        ncellsd = random.randint(0, tot // 2)
        ncells = random.choice((ncellsd, tot - ncellsd))
        ncells = min(max(1, ncells), tot)
        # Learnability: the generator translates the pattern's bbox by objh+3,
        # which equals "mirror across the rail" (and the verifier's "flush
        # against the rail") only when the pattern spans its whole band. objh is
        # not recoverable from the visible frame, so require the full span.
        if objh > 1:
            ncells = max(ncells, 2)
        while True:
            cells = random.sample(cands, ncells)
            rows = {r for r, _ in cells}
            if 0 in rows and (objh - 1) in rows:
                break
        mn = min(r for r, _ in cells)
        mx = max(r for r, _ in cells)
        for (i, j) in cells:
            canvi[i][j] = objc
            canvo[mn + mx - i + objh + 3][j] = objc
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
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    H, W = I.shape
    G = I.copy()
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = sorted(int(c) for c in cnt if c != bgc)

    # --- the frame: two full parallel rails + 4 inward stubs ---
    boxc, horizontal = None, True
    for c in others:
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if not all((int(r) in (r0, r1)) or (int(cc) in (c0, c1)) for r, cc in pts):
            continue
        rows_full = all(I[r0, j] == c for j in range(c0, c1 + 1)) and \
            all(I[r1, j] == c for j in range(c0, c1 + 1))
        cols_full = all(I[i, c0] == c for i in range(r0, r1 + 1)) and \
            all(I[i, c1] == c for i in range(r0, r1 + 1))
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
    if boxc is None:
        boxc = others[0]
        pts = np.argwhere(I == boxc)
        r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        horizontal = all(I[r0, j] == boxc for j in range(c0, c1 + 1))
    objc = [c for c in others if c != boxc][0]

    bx = np.argwhere(I == boxc)
    br0, br1 = int(bx[:, 0].min()), int(bx[:, 0].max())
    bc0, bc1 = int(bx[:, 1].min()), int(bx[:, 1].max())

    # Rail coordinates: a = axis perpendicular to the rails, b = along the rails.
    if horizontal:
        to_rc = lambda a, b: (a, b)
        rails = (br0, br1)
        flip_op = 27          # up<->down
        step_op = {-1: 20, 1: 21}
    else:
        to_rc = lambda a, b: (b, a)
        rails = (bc0, bc1)
        flip_op = 26          # left<->right
        step_op = {-1: 23, 1: 22}

    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    obj_ab = [((r, c) if horizontal else (c, r)) for r, c in obj]
    groups = [(rails[0], -1, [p for p in obj_ab if p[0] < rails[0]]),
              (rails[1], +1, [p for p in obj_ab if p[0] > rails[1]])]

    for rail, side, cells in groups:
        if not cells:
            continue
        a_lo = min(a for a, _ in cells)
        a_hi = max(a for a, _ in cells)
        b_lo = min(b for _, b in cells)
        b_hi = max(b for _, b in cells)
        far = a_lo if side < 0 else a_hi          # farthest line from the rail
        mirror_far = 2 * rail - far
        lo, hi = min(far, mirror_far), max(far, mirror_far)

        # 1. Mirror the pattern across its rail: flip the full rectangle centred on
        #    the rail line, spanning the pattern's extent along the rail (the whole
        #    rectangle is the intent: the rail maps onto itself, the frame interior
        #    there is plain background, the stubs lie outside the pattern's span).
        if horizontal:
            rect = [lo, b_lo, hi - lo, b_hi - b_lo]
            blk = G[lo:hi + 1, b_lo:b_hi + 1]
            G[lo:hi + 1, b_lo:b_hi + 1] = np.flipud(blk)
        else:
            rect = [b_lo, lo, b_hi - b_lo, hi - lo]
            blk = G[b_lo:b_hi + 1, lo:hi + 1]
            G[b_lo:b_hi + 1, lo:hi + 1] = np.fliplr(blk)
        ops.append(flip_op); sels.append(rect)
        cur = [(2 * rail - a, b) for a, b in cells]

        # 2. If the mirrored pattern does not sit one line inside the rail,
        #    slide it toward the rail.
        near = min(a for a, _ in cur) if side < 0 else max(a for a, _ in cur)
        target = rail + 2 if side < 0 else rail - 2
        k = abs(near - target)
        if k == 0:
            continue
        d = -1 if side < 0 else 1
        cur_rc = [to_rc(a, b) for a, b in cur]
        new_rc = [to_rc(a + d * k, b) for a, b in cur]
        if objc != 0:
            ops.append(step_op[d]); sels.append(sel_of(cur_rc))
            for _ in range(k - 1):
                ops.append(step_op[d]); sels.append(sel_of([]))
            hole = sorted(set(cur_rc) - set(new_rc))
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
        else:
            # colour 0 cannot be carried by ARCLE's Move (0 = "nothing")
            ops.append(bgc); sels.append(sel_of(sorted(set(cur_rc) - set(new_rc))))
            ops.append(0); sels.append(sel_of(sorted(set(new_rc) - set(cur_rc))))
        for r, c in cur_rc:
            G[r, c] = bgc
        for r, c in new_rc:
            G[r, c] = objc

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
