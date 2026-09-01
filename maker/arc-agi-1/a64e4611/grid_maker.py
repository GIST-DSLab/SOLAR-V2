"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback only
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for (r, c) in cells]}


# ----------------------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(round(a + (b - a) * diff_lb))
    hi = int(round(a + (b - a) * diff_ub))
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _detect(I):
    """Measure the drawn band structure from the INPUT only.

    The generator paints, on top of 2-colour noise, a solid `bgc` vertical band
    (columns locj..locj+dim-1, rows spi..h-1, i.e. always reaching the bottom edge)
    plus horizontal `bgc` bands shooting left and/or right from that column towards
    a side edge.  The rule is: fill the INTERIOR of every band with 3 -- interior
    meaning the band minus its 1-cell rim, where a rim that lies on a grid edge
    does not count (it is not a rim, the band is cut there).

    Returns the list of 3-rectangles (r0, c0, r1, c1) -- vertical band first.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # --- vertical band: consecutive columns whose bottom-anchored uniform run is long
    start = [0] * w
    for j in range(w):
        c = int(I[h - 1, j])
        r = h - 1
        while r - 1 >= 0 and int(I[r - 1, j]) == c:
            r -= 1
        start[j] = r
    lim = h // 2
    segs = []
    j = 0
    while j < w:
        if start[j] <= lim:
            col = int(I[h - 1, j])
            k = j
            while k + 1 < w and start[k + 1] <= lim and int(I[h - 1, k + 1]) == col:
                k += 1
            segs.append((j, k, col))
            j = k + 1
        else:
            j += 1
    cands = [s for s in segs if s[1] - s[0] + 1 >= 3]
    if not cands:
        raise ValueError("no vertical band")
    seg = max(cands, key=lambda s: (s[1] - s[0] + 1, -min(start[s[0]:s[1] + 1])))
    locj, lastj, bgc = seg[0], seg[1], seg[2]
    dim = lastj - locj + 1

    # top of the vertical band (row from which all its columns are bgc down to the bottom)
    top = h - 1
    r = h - 1
    while r - 1 >= 0 and bool(np.all(I[r - 1, locj:locj + dim] == bgc)):
        r -= 1
    top = r

    rects = []
    r0 = top + 1 if top > 0 else 0          # rim only where the band is not cut by the edge
    if r0 <= h - 1 and dim >= 3:
        rects.append((r0, locj + 1, h - 1, locj + dim - 2))

    # --- horizontal bands: rows that are solid bgc from the band out to a side edge
    def _groups(flags):
        out = []
        a = None
        for i, f in enumerate(flags):
            if f and a is None:
                a = i
            elif not f and a is not None:
                if i - a >= 3:
                    out.append((a, i - 1))
                a = None
        if a is not None and len(flags) - a >= 3:
            out.append((a, len(flags) - 1))
        return out

    left_flags = [bool(np.all(I[rr, 0:locj + 1] == bgc)) for rr in range(h)]
    right_flags = [bool(np.all(I[rr, locj + dim - 1:w] == bgc)) for rr in range(h)]

    for (a, b) in _groups(left_flags):
        t = a + 1 if a > 0 else 0
        if t <= b - 1:
            rects.append((t, 0, b - 1, locj + dim - 2))
    for (a, b) in _groups(right_flags):
        t = a + 1 if a > 0 else 0
        if t <= b - 1:
            rects.append((t, locj + 1, b - 1, w - 1))

    return rects


# ----------------------------------------------------------------------------- 1
VARIANTS = [{"sgns_key": "L"}, {"sgns_key": "R"}, {"sgns_key": "LR"}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sgns_key=None, **_kw) -> dict:
    if sgns_key is None:
        sgns_key = random.choice(["L", "R", "LR"])
    sgns = {"L": (-1,), "R": (1,), "LR": (-1, 1)}[sgns_key]

    hub = min(30, int(max_h))
    wub = min(30, int(max_w))
    hlb = min(18, hub)
    wlb = min(18, wub)

    last = None
    for _attempt in range(60):
        h = _unifint(diff_lb, diff_ub, (hlb, hub))
        w = _unifint(diff_lb, diff_ub, (wlb, wub))
        if h < 15 or w < 14:
            continue

        dim = random.randint(random.randint(3, 8), 8)
        if w - dim - 4 < 3:
            continue
        locj = random.randint(3, w - dim - 4)
        spi = random.choice((0, random.randint(3, h // 2))) if h // 2 >= 3 else 0

        opts = [spi]
        if spi + 3 <= h - 6:
            opts.append(random.randint(spi + 3, h - 6))
        startloc = random.choice(opts)
        hh_ub = min(8, h - startloc - 3)
        if hh_ub < 3:
            continue
        hh = random.randint(3, hh_ub)

        # ---- two-colour noise, then break every 3x3 monochrome block
        gi = np.full((h, w), noisec, dtype=int)
        n = h * w
        nbgc = _unifint(diff_lb, diff_ub, (int(0.4 * n), int(0.5 * n)))
        flat = gi.reshape(-1)
        for p in random.sample(range(n), nbgc):
            flat[p] = bgc
        addn, addb = [], []
        for r in range(h - 2):
            for c in range(w - 2):
                blk = gi[r:r + 3, c:c + 3]
                if bool(np.all(blk == bgc)):
                    addn.append((r + random.randint(0, 2), c + random.randint(0, 2)))
                elif bool(np.all(blk == noisec)):
                    addb.append((r + random.randint(0, 2), c + random.randint(0, 2)))
        for (r, c) in addn:
            gi[r, c] = noisec
        for (r, c) in addb:
            gi[r, c] = bgc
        go = gi.copy()

        # ---- vertical band + its interior
        gi[spi:h, locj:locj + dim] = bgc
        go[spi:h, locj:locj + dim] = bgc
        vt = spi + 1 if spi > 0 else spi
        go[vt:h, locj + 1:locj + dim - 1] = 3

        # ---- horizontal band(s)
        bands = [(sgns, startloc, hh)]
        if len(sgns) == 1 and _unifint(diff_lb, diff_ub, (0, 1)) == 1:
            opts2 = [spi]
            if spi + 3 <= h - 6:
                opts2.append(random.randint(spi + 3, h - 6))
            st2 = random.choice(opts2)
            ub2 = min(8, h - st2 - 3)
            if ub2 >= 3:
                bands.append(((-sgns[0],), st2, random.randint(3, ub2)))

        for (sg, st, hg) in bands:
            for sgn in sg:
                for ii in range(st, st + hg):
                    if sgn == -1:
                        c0, c1 = 0, locj + 1
                    else:
                        c0, c1 = locj, w
                    gi[ii, c0:c1] = bgc
                    row = go[ii, c0:c1]
                    go[ii, c0:c1] = np.where(row == 3, 3, bgc)
            for sgn in sg:
                t = st + 1 if st > 0 else st
                for ii in range(t, st + hg - 1):
                    if sgn == -1:
                        go[ii, 0:locj + dim - 1] = 3
                    else:
                        go[ii, locj + 1:w] = 3

        last = (gi, go)
        # ---- keep only instances whose structure is unambiguously readable from the input
        try:
            rects = _detect(gi)
        except Exception:
            continue
        pred = gi.copy()
        for (r0, c0, r1, c1) in rects:
            pred[r0:r1 + 1, c0:c1 + 1] = 3
        if not np.array_equal(pred, go):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}

    gi, go = last
    return {"input": gi.tolist(), "output": go.tolist()}


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # every parameter below is measured from I alone (band position, width, top row,
    # which sides the horizontal bands shoot to, how tall they are); 3 is the colour
    # the rule names, written as a constant.
    rects = _detect(I)

    ops, sels = [], []
    for (r0, c0, r1, c1) in rects:
        cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        ops.append(3)                 # Color3 over this band's interior
        sels.append(sel_of(cells))    # exact cells of the rectangle

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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
