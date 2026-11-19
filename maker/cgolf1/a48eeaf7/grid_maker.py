"""
ARC Task: a48eeaf7 (RE-ARC) — LLM-generated grid_maker
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
    bgc, sqc, dotc = random.sample(range(10), 3)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, dotc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if bgc is None or sqc is None or dotc is None:
        bgc, sqc, dotc = random.sample(range(10), 3)
    h = max(8, min(max_h, unifint(diff_lb, diff_ub, (8, min(30, max_h)))))
    w = max(8, min(max_w, unifint(diff_lb, diff_ub, (8, min(30, max_w)))))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(loci, loci + ih):
        for c in range(locj, locj + iw):
            gi[r][c] = sqc
            go[r][c] = sqc
    A = [(x, locj - 1) for x in range(loci, loci + ih)]
    Ap = [(x, random.randint(0, locj - 2)) for x in range(loci, loci + ih)]
    B = [(x, locj + iw) for x in range(loci, loci + ih)]
    Bp = [(x, random.randint(locj + iw + 1, w - 1)) for x in range(loci, loci + ih)]
    C = [(loci - 1, x) for x in range(locj, locj + iw)]
    Cp = [(random.randint(0, loci - 2), x) for x in range(locj, locj + iw)]
    D = [(loci + ih, x) for x in range(locj, locj + iw)]
    Dp = [(random.randint(loci + ih + 1, h - 1), x) for x in range(locj, locj + iw)]
    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    num = unifint(diff_lb, diff_ub, (1, len(srarr)))
    locs = set(random.sample(range(len(srarr)), num))
    for j, (s, d) in enumerate(zip(srarr, dearr)):
        if j in locs:
            gi[s[0]][s[1]] = dotc
            go[d[0]][d[1]] = dotc
    ncorn = unifint(diff_lb, diff_ub, (0, 4))

    def diag(sr, sc, dr, dc):
        out = []
        while 0 <= sr < h and 0 <= sc < w:
            out.append((sr, sc))
            sr += dr
            sc += dc
        return out

    corners = [
        ((loci - 1, locj - 1), (loci - 2, locj - 2, -1, -1)),
        ((loci - 1, locj + iw), (loci - 2, locj + iw + 1, -1, 1)),
        ((loci + ih, locj - 1), (loci + ih + 1, locj - 2, 1, -1)),
        ((loci + ih, locj + iw), (loci + ih + 1, locj + iw + 1, 1, 1)),
    ]
    for k in range(ncorn):
        (tr, tc), args = corners[k]
        go[tr][tc] = dotc
        sr, sc = random.choice(diag(*args))
        gi[sr][sc] = dotc
    rot = random.choice((0, 1, 2, 3))
    gi = np.rot90(np.array(gi), -rot).tolist()
    go = np.rot90(np.array(go), -rot).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (taken from the code-golf solution): the rule is GRAVITY toward the
    block, worked one side at a time.  The golf entry rotates the grid four
    times. Each time, in every column, it sorts the part before the block's edge
    so the dots sink against that edge.  So the unit here is not "one dot and
    its nearest outbox cell", which is how the maker reads it.  The unit is
    "every dot on one side of the block, falling together toward that side's
    edge line":
      pass 1  rows above the block    fall DOWN   to the row just above it
      pass 2  rows below the block    fall UP     to the row just below it
      pass 3  cols left of the block  slide RIGHT to the col just left of it
      pass 4  cols right of the block slide LEFT  to the col just right of it
    A diagonal (corner) dot takes part in two passes. First it falls onto the
    outbox row, then it slides along that row into the corner.  The maker sends
    it there in one trip instead.
    Each pass moves the whole group one cell per step.  The group is every dot
    on that side that has not yet reached the edge line, which can be read off
    the current grid.  The cells the group just left go back to background.
    Where the two readings agree: the colour roles, the block, and where each
    dot ends up are the same.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # colour roles: read them from the demonstrations when available
    bgc = sqc = dotc = None
    if examples:
        ei = np.asarray(examples[0][0], dtype=int)
        eo = np.asarray(examples[0][1], dtype=int)
        bgc = Counter(ei.flatten().tolist()).most_common(1)[0][0]
        moved = set(ei[ei != eo].tolist()) | set(eo[ei != eo].tolist())
        moved.discard(bgc)
        if len(moved) == 1:
            dotc = moved.pop()
            rest = [c for c in set(ei.flatten().tolist()) if c not in (bgc, dotc)]
            sqc = rest[0] if len(rest) == 1 else None
    present = set(I.flatten().tolist())
    if bgc not in present or sqc not in present or dotc not in present:
        cnt = Counter(I.flatten().tolist())
        bgc = cnt.most_common(1)[0][0]

        def is_rect(c):
            rs_, cs_ = np.where(I == c)
            return len(rs_) >= 4 and len(rs_) == (rs_.max() - rs_.min() + 1) * (cs_.max() - cs_.min() + 1)
        rects = [c for c in cnt if c != bgc and is_rect(c)]
        sqc = max(rects, key=lambda c: cnt[c])
        dotc = [c for c in cnt if c not in (bgc, sqc)][0]

    rs, cs = np.where(I == sqc)
    r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()

    G = I.copy()
    ops, sels = [], []

    # (which side, where it stops, step (dr, dc), Move op)
    passes = [
        (lambda r, c: r < r0, lambda r, c: r == r0 - 1, (1, 0), 21),
        (lambda r, c: r > r1, lambda r, c: r == r1 + 1, (-1, 0), 20),
        (lambda r, c: c < c0, lambda r, c: c == c0 - 1, (0, 1), 22),
        (lambda r, c: c > c1, lambda r, c: c == c1 + 1, (0, -1), 23),
    ]
    for side, stop, (dr, dc), mop in passes:
        while True:
            group = [(r, c) for r, c in zip(*np.where(G == dotc))
                     if side(r, c) and not stop(r, c)]
            if not group:
                break
            new = [(r + dr, c + dc) for r, c in group]
            left = sorted(set(group) - set(new))
            if dotc == 0:
                # a 0-coloured dot cannot ride a Move: paint the step instead
                ops.append(int(bgc)); sels.append(sel_of(left))
                ops.append(0); sels.append(sel_of(new))
            else:
                # the whole still-falling group steps one cell together
                ops.append(mop); sels.append(sel_of(group))
                if bgc != 0 and left:
                    ops.append(int(bgc)); sels.append(sel_of(left))
            for r, c in group:
                G[r, c] = bgc
            for r, c in new:
                G[r, c] = dotc

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])  # whole grid, intentional
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
                        f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a48eeaf7"
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
                                f"for task a48eeaf7"
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
                    f"Failed to build a complete episode for task a48eeaf7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a48eeaf7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
