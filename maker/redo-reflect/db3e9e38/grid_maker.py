"""
ARC Task: db3e9e38 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


ROTATIONS = [0, 1, 2, 3]          # number of CW quarter turns applied to the pair


# ----------------------------------------------------------------------------
# 1. colors  (+ pre-planned orientation per instance)
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]          # 8 is reserved by the task
    bgc = random.choice(cols)
    # fgc != 0 : 0 is "transparent" for CopyO/Paste, and the mirror half of the
    # pattern is produced with a real Copy/Paste/Flip.
    fgc = random.choice([c for c in cols if c != bgc and c != 0])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        ex = [{"rot": r} for r in ROTATIONS]
        ex += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = [dict(e) for e in ex] + [dict(random.choice(ex))]   # test uses a seen case
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTATIONS)

    max_h = max(3, min(30, int(max_h)))
    max_w = max(3, min(30, int(max_w)))

    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (3, max_w))
    barth = _unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    barth = max(1, min(barth, w - 2))
    loci = _unifint(diff_lb, diff_ub, (1, h - 2))
    locj = random.randint(1, w - barth - 1)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    # the bar: rows 0..loci, cols locj..locj+barth-1  (touches the top edge)
    for r in range(0, loci + 1):
        for c in range(locj, locj + barth):
            gi[r][c] = fgc

    # staircase: copy n is the bar shifted by (-n, +/- n*barth); colors alternate
    nmax = 2 * max(h, w) + 2
    for n in range(0, nmax):
        col = fgc if n % 2 == 0 else 8
        signs = (1,) if n == 0 else (1, -1)
        for sgn in signs:
            for r in range(-n, loci - n + 1):
                for c in range(locj + sgn * n * barth, locj + sgn * n * barth + barth):
                    if 0 <= r < h and 0 <= c < w:
                        go[r][c] = col

    for _ in range(rot % 4):                       # rotate both, CW
        gi = [list(x) for x in zip(*gi[::-1])]
        go = [list(x) for x in zip(*go[::-1])]

    return {"input": gi, "output": go}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    try:
        from maker.sel_helpers import sel_of
    except Exception:                              # pragma: no cover
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- find the canonical view: bar vertical, hanging from the top edge ----
    k = 0
    for kk in range(4):
        Ik = np.rot90(I, -kk)
        cs = np.argwhere(Ik != bgc)
        hk, wk = Ik.shape
        r0, r1 = int(cs[:, 0].min()), int(cs[:, 0].max())
        c0, c1 = int(cs[:, 1].min()), int(cs[:, 1].max())
        if r0 == 0 and r1 < hk - 1 and c0 > 0 and c1 < wk - 1:
            k = kk
            break

    Ic = np.rot90(I, -k)                            # canonical grid
    hc, wc = Ic.shape
    idx = np.rot90(np.arange(h * w).reshape(h, w), -k)

    def A(i, j):                                    # canonical -> actual coords
        v = int(idx[i, j])
        return (v // w, v % w)

    cs = np.argwhere(Ic != bgc)
    L = int(cs[:, 0].max())                         # bar occupies rows 0..L
    cc0, cc1 = int(cs[:, 1].min()), int(cs[:, 1].max())
    b = cc1 - cc0 + 1
    fgc = int(Ic[0, cc0])

    availL = cc0
    availR = wc - 1 - cc1
    side = 1 if availR >= availL else -1            # build the roomier side first
    M = min(availL, availR)

    ops, sels = [], []
    Gc = Ic.copy()                                  # python-side mirror of the grid

    # ---- paint the staircase on the roomier side, copy by copy, outward -----
    n = 1
    while True:
        bot = L - n
        if bot < 0:
            break
        j0 = cc0 + side * n * b
        j1 = cc1 + side * n * b
        lo, hi = max(0, j0), min(wc - 1, j1)
        if lo > hi:
            break
        color = fgc if n % 2 == 0 else 8
        cells = [(i, j) for i in range(0, bot + 1) for j in range(lo, hi + 1)]
        Gc[0:bot + 1, lo:hi + 1] = color
        ops.append(int(color))
        sels.append(sel_of([A(i, j) for i, j in cells]))   # exactly this copy's cells
        n += 1

    # ---- the other side IS the mirror image of this one: copy, paste, flip --
    if M >= 1 and L >= 1:
        if side == 1:
            s_lo, s_hi = cc1 + 1, cc1 + M
            d_lo, d_hi = cc0 - M, cc0 - 1
        else:
            s_lo, s_hi = cc0 - M, cc0 - 1
            d_lo, d_hi = cc1 + 1, cc1 + M

        src = [(i, j) for i in range(0, L) for j in range(s_lo, s_hi + 1)]
        dst = [(i, j) for i in range(0, L) for j in range(d_lo, d_hi + 1)]
        src_a = [A(i, j) for i, j in src]
        dst_a = [A(i, j) for i, j in dst]

        # CopyO: selection is exactly the rectangular strip beside the bar
        ops.append(29)
        sels.append(sel_of(src_a))

        # Paste at the mirrored strip's top-left corner
        tl = (min(r for r, _ in dst_a), min(c for _, c in dst_a))
        ops.append(30)
        sels.append(sel_of([tl]))

        # Flip the pasted strip about the bar's axis (skip only if the strip is
        # its own mirror, where the flip would change nothing at all)
        strip = Gc[0:L, s_lo:s_hi + 1]
        if not np.array_equal(strip, strip[:, ::-1]):
            flip_op = 26 if k % 2 == 0 else 27      # canonical left<->right
            ops.append(flip_op)
            sels.append(sel_of(dst_a))              # exactly the pasted rectangle

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(h) for c in range(w)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task db3e9e38"
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
                                f"for task db3e9e38"
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
                    f"Failed to build a complete episode for task db3e9e38 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"db3e9e38-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
