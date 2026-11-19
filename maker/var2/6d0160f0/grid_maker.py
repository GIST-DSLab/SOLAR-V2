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
def sample_colors(num_examples=None) -> dict:
    import random
    cols = [c for c in range(10) if c != 4]          # 4 is reserved for the marker
    bgc, linc = random.sample(cols, 2)               # background + frontier-line colour
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc, linc) -> dict:
    import numpy as np

    def _struct(arr):
        """Recover (bh, bw, linc, bgc) the way derive_operations does."""
        hi, wi = arr.shape
        uni_r = [r for r in range(hi) if len(set(arr[r].tolist())) == 1]
        uni_c = [c for c in range(wi) if len(set(arr[:, c].tolist())) == 1]
        if not uni_r or not uni_c:
            return None
        bh = None
        for r in uni_r:
            if (hi + 1) % (r + 1) == 0 and all(rr in uni_r for rr in range(r, hi, r + 1)):
                bh = r
                break
        bw = None
        for c in uni_c:
            if (wi + 1) % (c + 1) == 0 and all(cc in uni_c for cc in range(c, wi, c + 1)):
                bw = c
                break
        if bh is None or bw is None:
            return None
        lc = int(arr[bh, 0])
        inblk, ncell = {}, {}
        for r0 in range(0, hi, bh + 1):
            for c0 in range(0, wi, bw + 1):
                vals = [int(arr[r, c]) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
                for v in set(vals):
                    if v != lc and v != 4:
                        inblk[v] = inblk.get(v, 0) + 1
                for v in vals:
                    if v != lc and v != 4:
                        ncell[v] = ncell.get(v, 0) + 1
        if not inblk:
            return None
        ranked = sorted(inblk, key=lambda v: (inblk[v], ncell[v], -v), reverse=True)
        if len(ranked) > 1:
            a, b = ranked[0], ranked[1]
            if (inblk[a], ncell[a]) == (inblk[b], ncell[b]):
                return None                       # ambiguous background -> reject
        return (bh, bw, lc, ranked[0])

    cols = difference(interval(0, 10, 1), (4,))
    hcands = [x for x in range(2, 6) if x * x + x - 1 <= max_h] or [2]
    wcands = [x for x in range(2, 6) if x * x + x - 1 <= max_w] or [2]
    hub, wub = max(hcands), max(wcands)

    result = None
    for _attempt in range(400):
        h = unifint(diff_lb, diff_ub, (2, hub))
        w = unifint(diff_lb, diff_ub, (2, wub))
        nh, nw = h, w
        fullh = h * nh + nh - 1
        fullw = w * nw + nw - 1
        gi = canvas(bgc, (fullh, fullw))
        for iloc in range(h, fullh, h + 1):
            gi = fill(gi, linc, hfrontier((iloc, 0)))
        for jloc in range(w, fullw, w + 1):
            gi = fill(gi, linc, vfrontier((0, jloc)))
        noccs = unifint(diff_lb, diff_ub, (1, h * w))
        denseinds = asindices(canvas(-1, (h, w)))
        sparseinds = {(a * (h + 1), b * (w + 1)) for a, b in denseinds}
        locs = sample(totuple(sparseinds), noccs)
        trgtl = choice(locs)
        remlocs = remove(trgtl, locs)
        ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
        place = choice(totuple(denseinds))
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(cols, ncols)
        candss = totuple(remove(place, denseinds))
        trgrem = sample(candss, ntrgt)
        trgrem = {(choice(ccols), ij) for ij in trgrem}
        trgtobj = {(4, place)} | trgrem
        go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc),
                             multiply(place, increment((h, w)))))
        gi = paint(gi, shift(trgtobj, trgtl))
        for rl in remlocs:
            tlo = choice(totuple(ccols))
            ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
            inds = sample(totuple(denseinds), ncells)
            obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
            gi = paint(gi, shift(obj, rl))

        result = {'input': gi, 'output': go}
        got = _struct(np.array([list(row) for row in gi]))
        if got == (h, w, linc, bgc):
            return result
    return result


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # ---------- structure: full frontier lines cut the grid into equal blocks ----------
    uni_r = [r for r in range(hi) if len(set(I[r].tolist())) == 1]
    uni_c = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    bh = None
    for r in uni_r:
        if (hi + 1) % (r + 1) == 0 and all(rr in uni_r for rr in range(r, hi, r + 1)):
            bh = r
            break
    if bh is None:
        bh = uni_r[0]
    bw = None
    for c in uni_c:
        if (wi + 1) % (c + 1) == 0 and all(cc in uni_c for cc in range(c, wi, c + 1)):
            bw = c
            break
    if bw is None:
        bw = uni_c[0]
    linc = int(I[bh, 0])
    rstarts = list(range(0, hi, bh + 1))
    cstarts = list(range(0, wi, bw + 1))
    nbr, nbc = len(rstarts), len(cstarts)
    blocks = {}
    for bi, r0 in enumerate(rstarts):
        for bj, c0 in enumerate(cstarts):
            blocks[(bi, bj)] = [(r, c) for r in range(r0, r0 + bh)
                                for c in range(c0, c0 + bw)]

    # background = the colour living in the most blocks (ties -> the most cells)
    inblk, ncell = Counter(), Counter()
    for cells in blocks.values():
        vals = [int(I[r, c]) for r, c in cells]
        for v in set(vals):
            if v != linc and v != 4:
                inblk[v] += 1
        for v in vals:
            if v != linc and v != 4:
                ncell[v] += 1
    bgc = max(inblk, key=lambda v: (inblk[v], ncell[v], -v))

    # ---------- the unique 4 marks the tile; its place inside the tile names the target ----------
    p4 = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == 4][0]
    sb = (p4[0] // (bh + 1), p4[1] // (bw + 1))
    db = (p4[0] % (bh + 1), p4[1] % (bw + 1))
    src_cells = blocks[sb]
    sr, sc = rstarts[sb[0]], cstarts[sb[1]]
    tr, tc = rstarts[db[0]], cstarts[db[1]]
    dr, dc = tr - sr, tc - sc
    moving = (dr != 0 or dc != 0)

    dirty = {}
    for key, cells in blocks.items():
        d = [(r, c) for r, c in cells if int(I[r, c]) != bgc]
        if d:
            dirty[key] = d
    others = [k for k in blocks if k != sb and k != db]
    dirty_others = [k for k in others if k in dirty]

    # ---------- route choices: all read off the input, none off the answer ----------
    dest_dirty = dirty.get(db, []) if moving else []
    preclear = moving and len(dest_dirty) > 0 and 2 * len(dest_dirty) >= bh * bw
    vert_first = abs(dr) > abs(dc) or (abs(dr) == abs(dc) and linc % 2 == 0)
    wipe_first = (len(dirty_others) % 2 == 1)
    style = (bgc + linc) % 3            # 0 = tile by tile, 1 = row bands, 2 = column bands
    row_down = (2 * sb[0] <= nbr - 1)   # sweep away from the marked tile
    col_right = (2 * sb[1] <= nbc - 1)
    rows_major = (nbr >= nbc)
    tidy_before_fix = (p4[0] % 2 == 0)
    fix_asc = (p4[1] % 2 == 0)

    # ---------- carrying the marked tile to its target block ----------
    move_phase = []
    if preclear:
        # lay the target block back to background before the tile lands on it
        move_phase.append((bgc, sel_of(dest_dirty)))
    if moving:
        legs = [(21 if dr > 0 else 20, abs(dr)), (22 if dc > 0 else 23, abs(dc))]
        if not vert_first:
            legs = legs[::-1]
        first = True
        for mop, n in legs:
            for _ in range(n):
                move_phase.append((mop, sel_of(src_cells) if first else sel_of([])))
                first = False

    # the grab leaves the tile's own footprint at 0
    tidy = []
    if moving and bgc != 0:
        tidy.append((bgc, sel_of(src_cells)))

    # cells the transparent carry could not bring (colour 0) and line-coloured
    # cells, which the rule turns back into background
    fixes = {}
    for (r, c) in src_cells:
        v = int(I[r, c])
        want = bgc if v == linc else v
        if not moving or v != 0:
            have = v
        elif preclear:
            have = bgc
        else:
            have = int(I[r + dr, c + dc])
        if have != want:
            fixes.setdefault(want, []).append((r + dr, c + dc))
    fix_ops = [(col, sel_of(fixes[col])) for col in sorted(fixes, reverse=not fix_asc)]

    move_phase += (tidy + fix_ops) if tidy_before_fix else (fix_ops + tidy)

    # ---------- every other block goes back to plain background ----------
    def order_key(k):
        i, j = k
        ki = i if row_down else (nbr - 1 - i)
        kj = j if col_right else (nbc - 1 - j)
        return (ki, kj) if rows_major else (kj, ki)

    dirty_others = sorted(dirty_others, key=order_key)
    wipe = []
    if style == 0:
        for k in dirty_others:
            wipe.append((bgc, sel_of(dirty[k])))
    else:
        axis = 0 if style == 1 else 1
        bands = []
        for k in dirty_others:
            if k[axis] not in bands:
                bands.append(k[axis])
        for b in bands:
            cells = []
            for k in dirty_others:
                if k[axis] == b:
                    cells += dirty[k]
            wipe.append((bgc, sel_of(cells)))

    for op, sel in (wipe + move_phase) if wipe_first else (move_phase + wipe):
        ops.append(op)
        sels.append(sel)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])   # full-grid rectangle, Submit
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
