"""
ARC Task: 6b9890af (RE-ARC) — LLM-generated grid_maker
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

# How this reading differs from the maker:
# the golf solution works at the level of the whole frame, not single pixels.
# Its unit is "the frame, kept", and the pattern is upscaled into that frame.
# So this solution does two things in a different order from the maker:
#   1. It crops to the frame first, so the canvas becomes the frame.
#   2. It draws the upscaled pattern inside the frame, one colour layer at a time.
#      Each layer selects every fac x fac block whose source pixel has that colour.
# The maker instead paints one block per pixel and crops at the end.
# Both read the same rule: the frame is a 1-cell border, and fac comes from the
# interior size divided by the pattern size.


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + (b - a) * min(diff_lb, diff_ub)
    hi = a + (b - a) * max(diff_lb, diff_ub)
    return random.randint(int(round(lo)), int(round(hi)))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kw) -> dict:
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(range(10), 2)
    lim = min(max_h, max_w)
    oh = _unifint(diff_lb, diff_ub, (2, 5))
    ow = _unifint(diff_lb, diff_ub, (2, 5))
    oh = min(oh, max(2, (lim - 4) // 2))
    ow = min(ow, max(2, (lim - 4) // 2))
    h = _unifint(diff_lb, diff_ub, (2 * oh + 2, lim))
    w = _unifint(diff_lb, diff_ub, (2 * ow + 2, lim))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}

    def shp(s):
        rs = [a for a, _ in s]; cs = [b for _, b in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    obj = {random.choice(sorted(bounds))}
    while shp(obj) != (oh, ow):
        nb = set()
        for (a, b) in obj:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        nb.add((a + da, b + db))
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = _unifint(diff_lb, diff_ub, (1, max(1, maxfac)))
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    numc = _unifint(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numc)
    objc = {ij: random.choice(ccols) for ij in obj}
    gi = np.full((h, w), bgc, dtype=int)
    bh, bw = oh * fac + 2, ow * fac + 2
    li = random.randint(0, h - bh - oh)
    lj = random.randint(0, w - bw)
    gi[li, lj:lj + bw] = sqc; gi[li + bh - 1, lj:lj + bw] = sqc
    gi[li:li + bh, lj] = sqc; gi[li:li + bh, lj + bw - 1] = sqc
    pi = random.randint(li + bh, h - oh)
    pj = random.randint(0, w - ow)
    for (a, b), v in objc.items():
        gi[pi + a, pj + b] = v
    go = np.full((bh, bw), bgc, dtype=int)
    go[0, :] = sqc; go[-1, :] = sqc; go[:, 0] = sqc; go[:, -1] = sqc
    for (a, b), v in objc.items():
        go[1 + a * fac:1 + (a + 1) * fac, 1 + b * fac:1 + (b + 1) * fac] = v
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: shared by every grid in the episode; the examples settle it
    grids = [I] + [np.asarray(e[0], dtype=int) for e in (examples or [])]
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]

    # the frame: the same-colour component whose cells are exactly its bbox border
    seen = np.zeros((hi, wi), dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            st = [(r, c)]; cells = []
            while st:
                y, x = st.pop(); cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True; st.append((ny, nx))
            rs = [y for y, _ in cells]; cs = [x for _, x in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            border = {(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                      if y in (r0, r1) or x in (c0, c1)}
            if set(cells) == border:
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if best is None or area > best[0]:
                    best = (area, r0, c0, r1 - r0 + 1, c1 - c0 + 1, border)
    _, br, bc, bh, bw, bcells = best

    # the pattern: every other non-background cell
    pts = [(r, c) for r in range(hi) for c in range(wi)
           if I[r, c] != bgc and (r, c) not in bcells]
    pr = min(r for r, _ in pts); pc = min(c for _, c in pts)
    ph = max(r for r, _ in pts) - pr + 1
    pw = max(c for _, c in pts) - pc + 1
    fy = (bh - 2) // ph
    fx = (bw - 2) // pw

    ops, sels = [], []
    # 1. keep only the frame: crop the canvas to its full bounding rectangle
    ops.append(33); sels.append([br, bc, bh - 1, bw - 1])  # whole rectangle intended
    # 2. CropGrid is transparent; repaint any interior cell that read 0 (only if bgc != 0)
    if bgc != 0:
        inner = [(r, c) for r in range(1, bh - 1) for c in range(1, bw - 1)
                 if I[br + r, bc + c] == 0]
        if inner:
            ops.append(bgc); sels.append(sel_of(inner))
    # 3. draw the pattern upscaled into the frame interior, one colour layer at a time:
    #    each layer selects every fy x fx block whose source pixel has that colour
    by_col = {}
    for i in range(ph):
        for j in range(pw):
            v = int(I[pr + i, pc + j])
            if v == bgc:
                continue
            by_col.setdefault(v, []).append((i, j))
    for v in sorted(by_col):
        cells = [(1 + i * fy + a, 1 + j * fx + b)
                 for (i, j) in by_col[v] for a in range(fy) for b in range(fx)]
        ops.append(v); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
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
                                f"for task 6b9890af"
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
                    f"Failed to build a complete episode for task 6b9890af "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6b9890af-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
