"""
ARC Task: 6d0160f0 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    if bgc is None or linc is None:
        bgc, linc = random.sample(cols, 2)

    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        a = lo + (hi - lo) * diff_lb
        b = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, int(round(random.uniform(a, b)))))

    def fits(n, m):
        return n * n + n - 1 <= m

    hmax = max(k for k in range(2, 6) if fits(k, max_h)) if fits(2, max_h) else 2
    wmax = max(k for k in range(2, 6) if fits(k, max_w)) if fits(2, max_w) else 2
    h = unifint(2, hmax)
    w = unifint(2, wmax)
    nh, nw = h, w
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = [[bgc] * fullw for _ in range(fullh)]
    for r in range(h, fullh, h + 1):
        for c in range(fullw):
            gi[r][c] = linc
    for c in range(w, fullw, w + 1):
        for r in range(fullh):
            gi[r][c] = linc
    go = [row[:] for row in gi]

    denseinds = [(a, b) for a in range(h) for b in range(w)]
    sparseinds = [(a * (h + 1), b * (w + 1)) for a, b in denseinds]
    noccs = unifint(1, h * w)
    locs = random.sample(sparseinds, noccs)
    trgtl = random.choice(locs)
    remlocs = [l for l in locs if l != trgtl]
    ntrgt = unifint(1, h * w - 1)
    place = random.choice(denseinds)
    ncols = unifint(1, 9)
    ccols = random.sample(cols, ncols)
    candss = [d for d in denseinds if d != place]
    trgrem = random.sample(candss, min(ntrgt, len(candss)))
    trgtobj = [(4, place)] + [(random.choice(ccols), ij) for ij in trgrem]

    orr, occ = place[0] * (h + 1), place[1] * (w + 1)
    for v, (i, j) in trgtobj:
        if v != linc:
            go[i + orr][j + occ] = v
    for v, (i, j) in trgtobj:
        gi[i + trgtl[0]][j + trgtl[1]] = v

    for rl in remlocs:
        tlo = random.choice(ccols)
        ncells = unifint(1, h * w - 1)
        inds = random.sample(denseinds, ncells)
        pool = [c for c in ccols if c != tlo] if len(ccols) > 1 else ccols
        for (i, j) in inds:
            gi[i + rl[0]][j + rl[1]] = random.choice(pool)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ---- lattice: n blocks of size n per axis, fullsize = n*n + n - 1 ----
    def side(full):
        for n in range(1, 31):
            if n * n + n - 1 == full:
                return n
        return None
    bh, bw = side(hi), side(wi)
    rstarts = list(range(0, hi, bh + 1))
    cstarts = list(range(0, wi, bw + 1))
    linc = int(I[bh, 0])

    def block_cells_of(G, h_, w_):
        return [int(G[r][c]) for r in range(len(G)) for c in range(len(G[0]))
                if r % (h_ + 1) != h_ and c % (w_ + 1) != w_]

    # background: read from the demonstrations' outputs (all-but-one block empty)
    bgc = None
    if examples:
        try:
            EO = np.asarray(examples[0][1], dtype=int)
            eh, ew = side(EO.shape[0]), side(EO.shape[1])
            bgc = Counter(block_cells_of(EO, eh, ew)).most_common(1)[0][0]
        except Exception:
            bgc = None
    if bgc is None:
        bgc = Counter(block_cells_of(I, bh, bw)).most_common(1)[0][0]

    # ---- the 4 marks the source block; its in-block offset names the destination ----
    p4 = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == 4][0]
    sbr, sbc = p4[0] // (bh + 1), p4[1] // (bw + 1)
    dbr, dbc = p4[0] % (bh + 1), p4[1] % (bw + 1)
    sr, sc = rstarts[sbr], cstarts[sbc]
    tr, tc = rstarts[dbr], cstarts[dbc]
    dr, dc = tr - sr, tc - sc
    src_cells = [(r, c) for r in range(sr, sr + bh) for c in range(sc, sc + bw)]
    moving = (dr != 0 or dc != 0)

    # ---- 1. every output block other than the source starts empty: wipe them ----
    # (the destination block only needs wiping when bgc==0, since a bgc!=0 tile
    #  covers it entirely; its 0-cells get painted explicitly below)
    for br in range(len(rstarts)):
        for bc in range(len(cstarts)):
            if (br, bc) == (sbr, sbc):
                continue
            if moving and (br, bc) == (dbr, dbc) and bgc != 0:
                continue
            r0, c0 = rstarts[br], cstarts[bc]
            cells = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)
                     if int(I[r, c]) != bgc]
            if cells:
                ops.append(bgc); sels.append(sel_of(cells))

    # ---- 2. line-coloured marks are not part of the carried pattern ----
    lin_cells = [(r, c) for (r, c) in src_cells if int(I[r, c]) == linc]
    if lin_cells:
        ops.append(bgc); sels.append(sel_of(lin_cells))

    # ---- 3. carry the whole source tile to its destination block ----
    if moving:
        first = True
        vop = 21 if dr > 0 else 20
        for _ in range(abs(dr)):
            ops.append(vop); sels.append(sel_of(src_cells) if first else sel_of([]))
            first = False
        hop = 22 if dc > 0 else 23
        for _ in range(abs(dc)):
            ops.append(hop); sels.append(sel_of(src_cells) if first else sel_of([]))
            first = False
        if bgc != 0:
            ops.append(bgc); sels.append(sel_of(src_cells))   # vacated footprint
        # pattern cells of colour 0 are not carried by the transparent paste
        if bgc != 0:
            zero_cells = [(r + dr, c + dc) for (r, c) in src_cells if int(I[r, c]) == 0]
            if zero_cells:
                ops.append(0); sels.append(sel_of(zero_cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # whole-grid rectangle for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d0160f0"
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
                                f"for task 6d0160f0"
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
                    f"Failed to build a complete episode for task 6d0160f0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d0160f0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
