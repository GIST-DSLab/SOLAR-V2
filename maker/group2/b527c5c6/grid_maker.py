"""
ARC Task: b527c5c6 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    hmax = max(8, min(30, max_h))
    wmax = max(8, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (8, hmax))
    w = _unifint(diff_lb, diff_ub, (8, wmax))
    remcols = [c for c in range(10) if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    fullinds = {(i, j) for i in range(h) for j in range(w)}
    inds = set(fullinds)
    noccs = _unifint(diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = random.randint(3, random.randint(3, min(h, w) // 2 - 1))
        d2 = random.randint(d1 * 2 + 1, random.randint(d1 * 2 + 1, min(h, w) - 1))
        oh, ow = random.sample([d1, d2], 2)
        cands = [ij for ij in inds if 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1]
        if len(cands) == 0:
            continue
        loci, locj = random.choice(sorted(cands))
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if ow < oh:
            lrflag = True
            dc1 = {(i, locj) for i in range(loci + ow - 1, loci + oh - ow + 1)}
            dc2 = {(i, j + ow - 1) for i, j in dc1}
        else:
            lrflag = False
            dc1 = {(loci, j) for j in range(locj + oh - 1, locj + ow - oh + 1)}
            dc2 = {(i + oh - 1, j) for i, j in dc1}
        loc = random.choice(sorted(dc1 | dc2))
        sgn = -1 if loc in dc1 else 1
        direc = (sgn * (0 if lrflag else 1), sgn * (1 if lrflag else 0))
        ln = {(loc[0] + k * direc[0], loc[1] + k * direc[1]) for k in range(43)}
        s = min(oh, ow)
        lr0 = min(i for i, j in ln); lr1 = max(i for i, j in ln)
        lc0 = min(j for i, j in ln); lc1 = max(j for i, j in ln)
        shell = {(i, j) for i in range(lr0 - (s - 1), lr1 + s) for j in range(lc0 - (s - 1), lc1 + s)}
        sqc, dotc = random.sample(ccols, 2)
        goobj = {}
        for ij in (bd | shell) - ln:
            goobj[ij] = sqc
        for ij in ln:
            goobj[ij] = dotc
        goobj = {ij: v for ij, v in goobj.items() if ij in fullinds}
        goobji = set(goobj)
        if goobji.issubset(inds):
            succ += 1
            nb = set()
            for (i, j) in bd:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - goobji) - nb
            for (i, j) in bd:
                gi[i][j] = sqc
            gi[loc[0]][loc[1]] = dotc
            for (i, j), v in goobj.items():
                go[i][j] = v
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Generator keeps the outer frame of the input free of rectangles.
    bgc = int(I[0, 0])

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for sr in range(hi):
        for sc in range(wi):
            if seen[sr, sc] or I[sr, sc] == bgc:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)

    ops, sels = [], []
    for cells in comps:
        cnt = {}
        for r, c in cells:
            v = int(I[r, c])
            cnt[v] = cnt.get(v, 0) + 1
        dots = [(r, c) for r, c in cells if cnt[int(I[r, c])] == 1]
        if len(dots) != 1 or len(cnt) != 2:
            continue
        dr, dc = dots[0]
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        s = min(oh, ow)
        # The dot's cross-section: a strip of width 2s-1 centred on the dot,
        # lying along the rectangle edge that holds the dot (always inside the rect).
        if ow > oh:
            strip = [(dr, c) for c in range(dc - s + 1, dc + s)]
            step = (-1, 0) if dr == r0 else (1, 0)
            sbox = [dr, dc - s + 1, 0, 2 * s - 2]   # exact 1 x (2s-1) rectangle
        else:
            strip = [(r, dc) for r in range(dr - s + 1, dr + s)]
            step = (0, -1) if dc == c0 else (0, 1)
            sbox = [dr - s + 1, dc, 2 * s - 2, 0]   # exact (2s-1) x 1 rectangle
        zero_offsets = [(r - sbox[0], c - sbox[1]) for r, c in strip if I[r, c] == 0]
        # Copy the strip from the input once ...
        ops.append(28); sels.append(sbox)
        # ... and stamp it on every row/column from the rectangle edge outward.
        k = 1
        while True:
            pr = sbox[0] + k * step[0]
            pc = sbox[1] + k * step[1]
            if not (0 <= pr < hi and 0 <= pc < wi):
                break
            ops.append(30); sels.append([pr, pc, 0, 0])
            if zero_offsets:
                # Paste cannot write colour 0; paint those strip cells explicitly.
                ops.append(0)
                sels.append(sel_of([(pr + a, pc + b) for a, b in zero_offsets]))
            k += 1

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
                        f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
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
                                f"for task b527c5c6"
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
                    f"Failed to build a complete episode for task b527c5c6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b527c5c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
