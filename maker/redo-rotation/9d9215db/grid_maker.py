"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- variants
# The generator applies a random rot{0,90,180,270} to both grids, which places
# the "seed" quadrant in one of the four corners.  That is a discrete
# structural variant, so it is planned per-instance up front.
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    # The rule (mirror-symmetrize + extend each ring arm) is colour agnostic,
    # so only the background colour has to be fixed for the episode.
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if rot is None:
        rot = choice((0, 1, 2, 3))

    # a 90/270 rotation swaps the axes -> both sides must fit both bounds
    if rot in (1, 3):
        lim_h = lim_w = min(max_h, max_w)
    else:
        lim_h, lim_w = max_h, max_w
    hub = max(5, min(14, (lim_h - 1) // 2))
    wub = max(5, min(14, (lim_w - 1) // 2))

    h = unifint(diff_lb, diff_ub, (5, hub))
    w = unifint(diff_lb, diff_ub, (5, wub))
    h = h * 2 + 1
    w = w * 2 + 1

    remcols = remove(bgc, cols)
    ub = min(h, w) // 4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)
    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = sample(linsbase, nlines)
    colors = [choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)
    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2) for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: the input holds the seed quadrant of a set of concentric rectangular
    rings — a corner dot on the quadrant diagonal at (d, d) and, for some rings,
    a pair of arm seeds at (d, d+2) / (d+2, d) (measured from the anchor corner).
    Each arm seed shoots its line toward the centre, then the completed quadrant
    is mirrored across the vertical centre line and then across the horizontal
    centre line, giving the 4-fold symmetric ring picture.

    Trajectory: Color ops draw each arm line, then
    CopyO + Paste + FlipH  (mirror left<->right) and
    CopyO + Paste + FlipV  (mirror top<->bottom).
    """
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    cr, cc = (h - 1) // 2, (w - 1) // 2          # centre row / centre column
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    if not cells:
        return [34], [[0, 0, h - 1, w - 1]]

    # which corner quadrant holds the seed pattern
    top = max(r for r, c in cells) < cr
    left = max(c for r, c in cells) < cc

    def de(u, v):                                 # anchor coords -> grid coords
        return (u if top else h - 1 - u, v if left else w - 1 - v)

    G = I.copy()
    ops, sels = [], []

    ncells = {}
    for r, c in cells:
        u = r if top else h - 1 - r
        v = c if left else w - 1 - c
        ncells[(u, v)] = int(I[r, c])

    # ---- 1. shoot each arm seed toward the centre (one op per arm line) ----
    for (u, v) in sorted(ncells):
        col = ncells[(u, v)]
        if v == u + 2:                            # horizontal arm along row u
            tgt = [(u, vv) for vv in range(v + 2, cc + 1, 2)]
        elif u == v + 2:                          # vertical arm along column v
            tgt = [(uu, v) for uu in range(u + 2, cr + 1, 2)]
        else:
            continue                              # ring corner dot: nothing to draw
        pts = [de(a, b) for a, b in tgt]
        pts = [(r, c) for (r, c) in pts if G[r, c] != col]
        if pts:
            ops.append(col)
            sels.append(sel_of(pts))
            for r, c in pts:
                G[r, c] = col

    # ---- mirroring helper: duplicate a whole rectangle, then flip it ----
    def mirror_block(src_r, src_c, bh, bw, dst_r, dst_c, axis):
        sub = G[src_r:src_r + bh, src_c:src_c + bw].copy()
        # bbox selections here are exactly the full rectangles being copied /
        # pasted / flipped (background included), which is what these ops need.
        ops.append(29); sels.append([src_r, src_c, bh - 1, bw - 1])      # CopyO
        ops.append(30); sels.append([dst_r, dst_c, 0, 0])                # Paste
        dst = G[dst_r:dst_r + bh, dst_c:dst_c + bw]
        G[dst_r:dst_r + bh, dst_c:dst_c + bw] = np.where(sub != 0, sub, dst)
        # Paste is transparent for colour 0: restore any 0-coloured source cells
        zer = [(dst_r + i, dst_c + j) for i in range(bh) for j in range(bw)
               if sub[i, j] == 0 and G[dst_r + i, dst_c + j] != 0]
        if zer:
            ops.append(0); sels.append(sel_of(zer))
            for r, c in zer:
                G[r, c] = 0
        blk = G[dst_r:dst_r + bh, dst_c:dst_c + bw]
        flipped = np.flipud(blk) if axis == 0 else np.fliplr(blk)
        if not np.array_equal(blk, flipped):
            ops.append(27 if axis == 0 else 26)                          # FlipV / FlipH
            sels.append([dst_r, dst_c, bh - 1, bw - 1])
            G[dst_r:dst_r + bh, dst_c:dst_c + bw] = flipped

    # ---- 2. mirror across the vertical centre line (left <-> right) ----
    bh = cr + 1                                   # anchor rows 0..cr (centre row incl.)
    bw = cc                                       # half width, centre column excluded
    src_r = 0 if top else cr
    if left:
        src_c, dst_c = 0, cc + 1
    else:
        src_c, dst_c = cc + 1, 0
    mirror_block(src_r, src_c, bh, bw, src_r, dst_c, 1)

    # ---- 3. mirror across the horizontal centre line (top <-> bottom) ----
    if top:
        src_r2, dst_r2 = 0, cr + 1
    else:
        src_r2, dst_r2 = cr + 1, 0
    mirror_block(src_r2, 0, cr, w, dst_r2, 0, 0)

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
