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
    # bgc (canvas background) and linc (frontier-line colour) are the only random
    # colour roles of this generator that the rule depends on; object colours are
    # irrelevant to the rule (only the unique 4 marker matters).
    cols = [c for c in range(10) if c != 4]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    cols = [c for c in range(10) if c != 4]
    hopts = [x for x in range(2, 6) if x * x + x - 1 <= max_h] or [2]
    wopts = [x for x in range(2, 6) if x * x + x - 1 <= max_w] or [2]
    gi, go = None, None
    for _attempt in range(64):
        ha, hb = hopts[0], hopts[-1]
        h = random.randint(ha + int((hb - ha) * diff_lb), ha + int((hb - ha) * diff_ub))
        h = max(ha, min(hb, h))
        wa, wb = wopts[0], wopts[-1]
        w = random.randint(wa + int((wb - wa) * diff_lb), wa + int((wb - wa) * diff_ub))
        w = max(wa, min(wb, w))
        fullh = h * h + h - 1
        fullw = w * w + w - 1
        g = [[bgc] * fullw for _ in range(fullh)]
        for iloc in range(h, fullh, h + 1):
            for j in range(fullw):
                g[iloc][j] = linc
        for jloc in range(w, fullw, w + 1):
            for i in range(fullh):
                g[i][jloc] = linc
        o = [row[:] for row in g]
        dense = [(a, b) for a in range(h) for b in range(w)]
        sparse = [(a * (h + 1), b * (w + 1)) for a, b in dense]
        cap = max(1, (h * w - 1) // 2)  # keeps bgc identifiable inside every block
        noccs = random.randint(1 + int((h * w - 1) * diff_lb), 1 + int((h * w - 1) * diff_ub))
        noccs = max(1, min(h * w, noccs))
        locs = random.sample(sparse, noccs)
        trgtl = random.choice(locs)
        remlocs = [l for l in locs if l != trgtl]
        ntrgt = random.randint(1 + int((cap - 1) * diff_lb), 1 + int((cap - 1) * diff_ub))
        ntrgt = max(1, min(cap, ntrgt))
        place = random.choice(dense)
        ncols = random.randint(1 + int(8 * diff_lb), 1 + int(8 * diff_ub))
        ncols = max(1, min(9, ncols))
        ccols = random.sample(cols, ncols)
        cands = [ij for ij in dense if ij != place]
        trgrem = random.sample(cands, min(ntrgt, len(cands)))
        tobj = [(4, place)] + [(random.choice(ccols), ij) for ij in trgrem]
        for col, (a, b) in tobj:
            g[trgtl[0] + a][trgtl[1] + b] = col
        orow, ocol = place[0] * (h + 1), place[1] * (w + 1)
        for col, (a, b) in tobj:
            if col != linc:  # line-coloured cells of the object are dropped in the output
                o[orow + a][ocol + b] = col
        for rl in remlocs:
            ncells = random.randint(1 + int((cap - 1) * diff_lb), 1 + int((cap - 1) * diff_ub))
            ncells = max(1, min(cap, ncells))
            inds = random.sample(dense, ncells)
            tlo = random.choice(ccols)
            pool = [c for c in ccols if c != tlo] or ccols
            for (a, b) in inds:
                g[rl[0] + a][rl[1] + b] = random.choice(pool)
        gi, go = g, o
        # well-posedness: bgc present in every block and strictly the commonest cell colour
        cells = [g[r][c] for r in range(fullh) for c in range(fullw)
                 if r % (h + 1) != h and c % (w + 1) != w]
        cnt = Counter(cells)
        blocks_ok = True
        for a0 in range(0, fullh, h + 1):
            for b0 in range(0, fullw, w + 1):
                if bgc not in [g[r][c] for r in range(a0, a0 + h) for c in range(b0, b0 + w)]:
                    blocks_ok = False
        if blocks_ok and all(cnt[bgc] > cnt[c] for c in cnt if c != bgc):
            break
    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ---- structure: full frontier rows/cols cut the grid into equal blocks ----
    line_rows = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    line_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bh, bw = line_rows[0], line_cols[0]          # block height / width
    linc = int(I[line_rows[0], 0])
    rstarts = list(range(0, hi, bh + 1))
    cstarts = list(range(0, wi, bw + 1))
    nbr, nbc = len(rstarts), len(cstarts)

    # background = commonest colour among block (non-line) cells
    block_vals = [int(I[r, c]) for r in range(hi) for c in range(wi)
                  if r % (bh + 1) != bh and c % (bw + 1) != bw]
    bgc = Counter(block_vals).most_common(1)[0][0]

    # ---- the unique 4 marks the source block; its position inside that block
    #      names the destination block ----
    p4 = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == 4][0]
    sbr, sbc = p4[0] // (bh + 1), p4[1] // (bw + 1)
    dbr, dbc = p4[0] % (bh + 1), p4[1] % (bw + 1)
    sr, sc = rstarts[sbr], cstarts[sbc]
    tr, tc = rstarts[dbr], cstarts[dbc]
    dr, dc = tr - sr, tc - sc
    # full block rectangle (background included) — this whole tile is what travels
    src_cells = [(r, c) for r in range(sr, sr + bh) for c in range(sc, sc + bw)]

    # ---- slide the marked tile to its destination block ----
    if dr != 0 or dc != 0:
        first = True
        vop = 21 if dr > 0 else 20
        for _ in range(abs(dr)):
            ops.append(vop)
            sels.append(sel_of(src_cells) if first else sel_of([]))
            first = False
        hop = 22 if dc > 0 else 23
        for _ in range(abs(dc)):
            ops.append(hop)
            sels.append(sel_of(src_cells) if first else sel_of([]))
            first = False
        # the tile's original footprint is left at 0 by the grab
        if bgc != 0:
            ops.append(bgc)
            sels.append(sel_of(src_cells))

    # ---- destination touch-ups: cells the transparent paste could not carry
    #      (source value 0) and line-coloured cells, which become background ----
    fixes = {}
    for (r, c) in src_cells:
        v = int(I[r, c])
        want = bgc if v == linc else v
        if dr != 0 or dc != 0:
            have = v if v != 0 else int(I[r + dr, c + dc])
        else:
            have = v
        if have != want:
            fixes.setdefault(want, []).append((r + dr, c + dc))
    for col in sorted(fixes):
        ops.append(col)
        sels.append(sel_of(fixes[col]))

    # ---- every other block is wiped back to background, block by block ----
    for br in range(nbr):
        for bc in range(nbc):
            if (br, bc) == (sbr, sbc) or (br, bc) == (dbr, dbc):
                continue
            r0, c0 = rstarts[br], cstarts[bc]
            cells = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)
                     if int(I[r, c]) != bgc]
            if cells:
                ops.append(bgc)
                sels.append(sel_of(cells))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
