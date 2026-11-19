"""
ARC Task: 7e0986d6 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    # Only the background has to be shared across the episode: the rule is
    # "small blobs vanish, real blocks become solid rectangles" — purely a
    # size/shape rule, so the block and noise colours may vary per instance.
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = remove(bgc, cols)
    nsqcols = unifint(diff_lb, diff_ub, (1, 5))
    sqcols = sample(remcols, nsqcols)
    remcols = difference(remcols, sqcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(2, min(7, h))
        ow = randint(2, min(7, w))
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = choice(sqcols)
            go = fill(go, col, sq)
    gi = tuple(e for e in go)
    namt = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 9)))
    cands = asindices(gi)
    for k in range(namt):
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        col = gi[loc[0]][loc[1]]
        torem = neighbors(loc) & ofcolor(gi, col)
        cands = cands - torem
        noisec = choice(noisecols)
        gi = fill(gi, noisec, {loc})
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the demonstrations): every blob of 1-2 cells is dirt and goes
    back to the background; every bigger blob is a rectangle that lost a few
    cells to that dirt, so it is restored to its full bounding rectangle.

    The ROUTE is not fixed: where the sweep starts, which way it runs, whether
    the dirt is wiped before or after the rectangles are laid down, whether the
    dirt goes in one wipe or colour by colour, and whether a repeated rectangle
    is re-drawn or stamped from the first one -- all of these are decided from
    the input picture alone (largest block's position, grid shape, counts,
    background colour), so one instance always gives one route and different
    instances give different routes.
    """
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    h, w = A.shape

    # ---- background: the colour that dominates this picture and every
    #      demonstration picture (they all share one background) ----
    tally = Counter(A.flatten().tolist())
    if examples:
        for pair in examples:
            try:
                ei = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            tally.update(ei.flatten().tolist())
    bgc = int(sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])

    # ---- single-colour 4-connected blobs of the INPUT ----
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(A[r, c])
            seen[r, c] = True
            st = [(r, c)]
            cells = []
            while st:
                a, b = st.pop()
                cells.append((int(a), int(b)))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and A[na, nb] == col:
                        seen[na, nb] = True
                        st.append((na, nb))
            comps.append((col, cells))

    speckles = [(col, cells) for col, cells in comps if col != bgc and len(cells) < 3]
    blocks = []
    for col, cells in comps:
        if col == bgc or len(cells) < 3:
            continue
        rs = [a for a, _ in cells]
        cs = [b for _, b in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        blocks.append({
            "col": int(col), "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "rect": [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)],
        })

    speckle_cells = [cell for _, cl in speckles for cell in cl]
    covered = set()
    for b in blocks:
        covered.update(b["rect"])

    # ---- route choices, all read off the input ----
    if blocks:
        big = max(blocks, key=lambda b: ((b["r1"] - b["r0"] + 1) * (b["c1"] - b["c0"] + 1),
                                         -b["r0"], -b["c0"]))
        downward = (big["r0"] + big["r1"]) < (h - 1)     # biggest block sits high -> sweep down
        rightward = (big["c0"] + big["c1"]) < (w - 1)    # ... sits left -> sweep right
        rows_major = h >= w                              # sweep along the taller/wider axis
        def okey(b):
            rk = b["r0"] if downward else -b["r1"]
            ck = b["c0"] if rightward else -b["c1"]
            return (rk, ck) if rows_major else (ck, rk)
        order = sorted(blocks, key=okey)
    else:
        order = []

    spk_cols = sorted({col for col, _ in speckles})
    draw_first = ((bgc + len(blocks)) % 2 == 1)          # tidy up first, or draw first
    gmode = (len(spk_cols) + len(blocks)) % 3            # one wipe / colour up / colour down
    use_stamp = ((h + w + len(blocks)) % 2 == 0)         # stamp repeats, or redraw each

    # ---- which rectangles actually need laying down at all ----
    pre = A.copy()
    if not draw_first:
        for (a, b) in speckle_cells:
            pre[a, b] = bgc
    needs = [any(pre[a, b] != blk["col"] for (a, b) in blk["rect"]) for blk in order]

    cur = A.copy()
    ops, sels = [], []

    def wipe_dirt(skip_covered):
        cells = [(a, b) for (a, b) in speckle_cells
                 if cur[a, b] != bgc and not (skip_covered and (a, b) in covered)]
        if not cells:
            return
        if gmode == 0:
            groups = [sorted(cells)]
        else:
            bycol = {}
            for (a, b) in cells:
                bycol.setdefault(int(cur[a, b]), []).append((a, b))
            keys = sorted(bycol, reverse=(gmode == 2))
            groups = [sorted(bycol[k]) for k in keys]
        for g in groups:
            ops.append(int(bgc))          # Color<bgc> on exactly the dirt cells
            sels.append(sel_of(g))
            for (a, b) in g:
                cur[a, b] = bgc

    def lay_blocks():
        clip = None
        for i, blk in enumerate(order):
            if not needs[i]:
                continue
            col = blk["col"]
            rect = blk["rect"]
            bh = blk["r1"] - blk["r0"] + 1
            bw = blk["c1"] - blk["c0"] + 1
            sig = (col, bh, bw)
            # colour 0 never travels through the clipboard -> always paint it
            if use_stamp and col != 0 and clip == sig:
                ops.append(30)                                    # Paste the twin
                sels.append(sel_of([(blk["r0"], blk["c0"])]))
            else:
                ops.append(int(col))                              # Color<col>, whole rectangle
                sels.append(sel_of(rect))                         # (selection IS that full rect)
                if use_stamp and col != 0 and any(
                        needs[j] and (order[j]["col"],
                                      order[j]["r1"] - order[j]["r0"] + 1,
                                      order[j]["c1"] - order[j]["c0"] + 1) == sig
                        for j in range(i + 1, len(order))):
                    ops.append(29)                                # CopyO: keep this one to stamp
                    sels.append(sel_of(rect))
                    clip = sig
            for (a, b) in rect:
                cur[a, b] = col

    if draw_first:
        lay_blocks()
        wipe_dirt(skip_covered=True)     # rectangles already buried the dirt on them
    else:
        wipe_dirt(skip_covered=False)    # thorough tidy, then draw over the clean sheet
        lay_blocks()

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
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
                                f"for task 7e0986d6"
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
                    f"Failed to build a complete episode for task 7e0986d6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7e0986d6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
