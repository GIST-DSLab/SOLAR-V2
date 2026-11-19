"""
ARC Task: 00d62c1b (RE-ARC) — LLM-generated grid_maker
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

# Reading taken from the code-golf entry. It differs from the maker's reading.
# The maker treats each enclosed background pocket as its own unit and fills
# pockets one by one with FloodFill4. The golf entry treats the OUTSIDE as the
# unit: it marks every background cell reachable from the border, and every
# other background cell becomes 4.
# The ops follow that reading:
#   1. Color4 on "all cells of the background colour". Every background cell
#      becomes a candidate interior.
#   2. Walk the border. Each border cell that is still part of an un-restored
#      outside region gets one FloodFill(bgc), which gives that region back
#      its background colour.
# Whatever stays 4 is exactly the enclosed area. Both selections can be named
# from I: one colour class, and single border seed cells.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))

    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    nblocks = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    succ, tr, maxtr = 0, 0, 5 * nblocks
    inds = {(i, j) for i in range(h) for j in range(w)}
    while succ < nblocks and tr < maxtr:
        tr += 1
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bx = set()
        for j in range(locj, locj + ow):
            bx.add((loci, j)); bx.add((loci + oh - 1, j))
        for i in range(loci, loci + oh):
            bx.add((i, locj)); bx.add((i, locj + ow - 1))
        crn = [(loci, locj), (loci, locj + ow - 1),
               (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)]
        bx -= set(random.sample(crn, random.randint(0, 4)))
        if bx.issubset(inds) and len(inds - bx) > (h * w) // 2 + 1:
            for (i, j) in bx:
                gi[i][j] = fgc
            succ += 1
            inds -= bx
    fcount = sum(row.count(fgc) for row in gi)
    maxnnoise = max(0, (h * w) // 2 - 1 - fcount)
    namt = unifint(diff_lb, diff_ub, (0, maxnnoise))
    namt = min(namt, len(inds))
    for (i, j) in random.sample(sorted(inds), namt):
        gi[i][j] = fgc

    # Output: fill the background components that touch no border with 4.
    go = [row[:] for row in gi]
    seen = set()
    for si in range(h):
        for sj in range(w):
            if (si, sj) in seen or gi[si][sj] != bgc:
                continue
            seen.add((si, sj))
            st, comp, bord = [(si, sj)], [], False
            while st:
                r, c = st.pop()
                comp.append((r, c))
                if r in (0, h - 1) or c in (0, w - 1):
                    bord = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (r + dr, c + dc)
                    if 0 <= n[0] < h and 0 <= n[1] < w and n not in seen and gi[n[0]][n[1]] == bgc:
                        seen.add(n)
                        st.append(n)
            if not bord:
                for (r, c) in comp:
                    go[r][c] = 4
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    # Background: the generator fills the canvas with bgc and keeps fgc under half the grid.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    bg_cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] == bgc]

    # 4-connected background components of I (Python-side analysis only).
    seen, comps = set(), []
    for s in bg_cells:
        if s in seen:
            continue
        seen.add(s)
        st, comp = [s], []
        while st:
            r, c = st.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (r + dr, c + dc)
                if 0 <= n[0] < h and 0 <= n[1] < w and n not in seen and I[n] == bgc:
                    seen.add(n)
                    st.append(n)
        comps.append(set(comp))

    # Walk the border clockwise. Take one seed per outside (border-touching) region.
    border = ([(0, c) for c in range(w)] +
              [(r, w - 1) for r in range(1, h)] +
              [(h - 1, c) for c in range(w - 2, -1, -1)] +
              [(r, 0) for r in range(h - 2, 0, -1)])
    outer_seeds, outside = [], set()
    for b in border:
        if I[b] == bgc and b not in outside:
            comp = next(cc for cc in comps if b in cc)
            outside |= comp
            outer_seeds.append(b)

    ops, sels = [], []
    # Only act if some background is enclosed. Otherwise Color4 followed by the
    # restores would bring the grid back to I, which is a redundant cycle.
    if len(outside) < len(bg_cells):
        # 1. Every background-coloured cell becomes 4.
        ops.append(4); sels.append(sel_of(bg_cells))
        # 2. Give each border-reachable region its background colour back.
        for s in outer_seeds:
            ops.append(10 + bgc); sels.append(sel_of([s]))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # submit the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task 00d62c1b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 00d62c1b"
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
                                f"for task 00d62c1b"
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
                    f"Failed to build a complete episode for task 00d62c1b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"00d62c1b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
