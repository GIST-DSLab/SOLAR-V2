"""
ARC Task: ef135b50 (RE-ARC) — LLM-generated grid_maker
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
    # Colour 9 is reserved for the output fill, so the generator never uses it.
    cols = [c for c in range(10) if c != 9]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, len(remcols))
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "ccols": ccols}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = random.uniform(lo, hi)
    return int(max(a, min(b, round(v))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = [c for c in range(10) if c != 9]
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        rem = [c for c in cols if c != bgc]
        ccols = random.sample(rem, random.randint(1, len(rem)))
    hub = max(8, min(30, max_h))
    wub = max(8, min(30, max_w))
    while True:
        h = _unifint(diff_lb, diff_ub, (8, hub))
        w = _unifint(diff_lb, diff_ub, (8, wub))
        gi = [[bgc] * w for _ in range(h)]
        nsq = _unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 30)))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        inds = {(i, j) for i in range(h) for j in range(w)}
        pats = []
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = random.randint(1, h // 3 * 2)
            ow = random.randint(1, w // 3 * 2)
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loci, locj = random.choice(cands)
            bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
            if bd.issubset(inds):
                succ += 1
                nb = set()
                for (i, j) in bd:
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            nb.add((i + di, j + dj))
                inds = (inds - bd) - nb
                col = random.choice(ccols)
                for (i, j) in bd:
                    gi[i][j] = col
                pats.append((loci, locj, loci + oh - 1, locj + ow - 1))
        res = set()
        for a in range(len(pats)):
            for b in range(len(pats)):
                if a == b:
                    continue
                u1, l1, d1, r1 = pats[a]
                u2, l2, d2, r2 = pats[b]
                if max(u1, u2) <= min(d1, d2):  # hmatching
                    um = max(u1, u2)
                    bm = min(d1, d2)
                    lm = min(r1, r2) + 1
                    rm = max(l1, l2) - 1
                    for i in range(min(um, bm), max(um, bm) + 1):
                        for j in range(min(lm, rm), max(lm, rm) + 1):
                            res.add((i, j))
        go = [row[:] for row in gi]
        changed = False
        for (i, j) in res:
            if 0 < i < h - 1 and 0 < j < w - 1 and gi[i][j] == bgc:
                go[i][j] = 9
                changed = True
        if changed:
            return {"input": gi, "output": go}


def _background_colour(I):
    """Background = largest-bbox connected same-colour component that is NOT a
    solid rectangle (every placed block is a solid rectangle by construction)."""
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    bgc = int(I[0, 0])
    best = -1
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0]:
                continue
            col = int(I[r0, c0])
            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            bh = max(rs) - min(rs) + 1
            bw = max(cs) - min(cs) + 1
            if len(cells) == bh * bw:
                continue
            if bh * bw > best:
                best = bh * bw
                bgc = col
    return bgc


def derive_operations(I, O, examples=None):
    """
    Reading (per row, as in the reference): in each interior row, a background
    cell turns 9 when there is a block to its left AND a block to its right in
    that same row. So the unit of work is one row's background gap between two
    blocks. Each gap gets its own Color9, rows top to bottom, gaps left to right.
    Border rows are exempt. Only I is read; O supplies just the Submit shape.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    bgc = _background_colour(I)

    ops, sels = [], []
    for r in range(1, h - 1):
        occ = [c for c in range(w) if I[r, c] != bgc]
        if len(occ) < 2:
            continue
        lo, hi = occ[0], occ[-1]
        c = lo
        while c <= hi:
            if I[r, c] == bgc:
                s = c
                while c <= hi and I[r, c] == bgc:
                    c += 1
                # background gap in this row, bounded by a block on each side
                ops.append(9)
                sels.append(sel_of([(r, cc) for cc in range(s, c)]))
            else:
                c += 1

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
                        f"num_examples+1 ({num_examples + 1}) for task ef135b50"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ef135b50"
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
                                f"for task ef135b50"
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
                    f"Failed to build a complete episode for task ef135b50 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ef135b50-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
