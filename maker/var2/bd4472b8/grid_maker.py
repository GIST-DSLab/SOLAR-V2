"""
ARC Task: bd4472b8 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- variants ---
# The only discrete structural variant this task has is the quarter turn the
# whole picture got (header strip on top / right / bottom / left).
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    """Episode palette.  The rule itself is colour-blind (the stripe colours are
    read off the header strip of each instance), so only the two structural
    roles are pinned: the background field and the separator line.  `0` is an
    ordinary colour here - the route must cope with it."""
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rot=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = int(a + (b - a) * lb)
        hi = int(a + (b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    # canonical grid is (h + 2) x w : header strip, separator line, plain field
    wmax_a, hmax_a = min(8, max_w), min(28, max_h - 2)      # rot 0 / 180
    wmax_b, hmax_b = min(8, max_h), min(28, max_w - 2)      # rot 90 / 270
    ok_a = wmax_a >= 2 and hmax_a >= 1
    ok_b = wmax_b >= 2 and hmax_b >= 1

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    upright = rot in (0, 2)
    if upright and not ok_a and ok_b:
        upright = False
        rot = random.choice([1, 3])
    if (not upright) and not ok_b and ok_a:
        upright = True
        rot = random.choice([0, 2])

    if upright and ok_a:
        w = unifint(diff_lb, diff_ub, (2, wmax_a))
        h = unifint(diff_lb, diff_ub, (1, hmax_a))
    elif ok_b:
        w = unifint(diff_lb, diff_ub, (2, wmax_b))
        h = unifint(diff_lb, diff_ub, (1, hmax_b))
    else:
        upright, rot, w, h = True, 0, 2, 1

    ccols = random.sample([c for c in range(10) if c != bgc], w)

    gi = [list(ccols), [linc] * w] + [[bgc] * w for _ in range(h)]
    go = [list(ccols), [linc] * w] + [[ccols[i % w]] * w for i in range(h)]

    gi = np.rot90(np.array(gi, dtype=int), -rot)
    go = np.rot90(np.array(go, dtype=int), -rot)
    return {"input": [[int(v) for v in row] for row in gi.tolist()],
            "output": [[int(v) for v in row] for row in go.tolist()]}


def derive_operations(I, O):
    """
    Rule: the header strip of w colours is reflected across the diagonal (a
    quarter turn of the strip), each colour then spreads along its own stripe
    line, and the resulting w x w block repeats away from the header until the
    field is full.

    The rule is fixed; the ROUTE is not.  Four honest solver decisions are taken,
    each read off the input itself (so one instance always gives one route, and
    the variety comes from the instances differing):

      * reflect the strip with Copy/Paste/Rotate, or lay the first block's
        stripes down by hand          -> parity of the header's first colour
      * lay the strip on the near or the far edge of that square (which decides
        which side the seed column ends up on)
                                       -> parity of the header's last colour
      * sweep the stripes outward from the header, or back towards it
                                       -> parity of the number of stripe lines
      * repeat the finished block by copying it, or paint every repeated stripe
        line one at a time            -> which quarter turn the picture got

    Nothing is read from O.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # ---- find the quarter turn that puts the header strip on top ------------
    def canon_bg(G, strict=True):
        Hc, Wc = G.shape
        if Hc < 3 or Wc < 2 or Wc > 8:
            return None
        head = G[0].tolist()
        if strict:
            if len(set(head)) != Wc:
                return None
        elif len(set(head)) < 2:
            return None
        if len(set(G[1].tolist())) != 1:            # the separator line
            return None
        rest = set(G[2:].flatten().tolist())        # the plain field
        if len(rest) != 1:
            return None
        bg = rest.pop()
        if bg == int(G[1][0]):
            return None
        if strict and bg in set(head):
            return None
        return int(bg)

    k = Gc = bgc = None
    for strict in (True, False):
        for kk in range(4):
            G = np.rot90(I, kk)
            bg = canon_bg(G, strict)
            if bg is not None:
                k, Gc, bgc = kk, G, bg
                break
        if k is not None:
            break

    Hc, Wc = Gc.shape
    w = Wc
    nstripe = Hc - 2
    ccols = [int(x) for x in Gc[0].tolist()]

    def M(r, c):                       # canonical -> actual (the quarter turn)
        if k == 0:
            return (r, c)
        if k == 1:
            return (c, Hc - 1 - r)
        if k == 2:
            return (Hc - 1 - r, Wc - 1 - c)
        return (Wc - 1 - c, r)

    def row_cells(r):
        return [M(r, c) for c in range(w)]

    def block_cells(r0, nrows, ncols):
        return [M(r, c) for r in range(r0, r0 + nrows) for c in range(ncols)]

    def origin_of(cells):
        return (min(p[0] for p in cells), min(p[1] for p in cells))

    def stripe_color(r):
        return ccols[(r - 2) % w]

    cur = I.copy()
    ops, sels = [], []

    def paint(color, cells):
        """Colour these cells, skipping any that already hold that colour;
        emit nothing at all when there is nothing left to change."""
        todo = [p for p in cells if int(cur[p[0], p[1]]) != color]
        if not todo:
            return
        ops.append(int(color))
        sels.append(sel_of(todo))
        for r, c in todo:
            cur[r, c] = color

    # ---- the four route decisions, all read off the input -------------------
    s = min(w, nstripe)                       # side of the square the turn uses
    nblocks = nstripe // w
    use_reflect = (s >= 2) and (ccols[0] % 2 == 0)
    seed_far = (ccols[-1] % 2 == 0)
    reverse = (nstripe % 2 == 1)
    use_tile = (nblocks >= 2) and (k % 2 == 0)

    def order(seq):
        seq = list(seq)
        return list(reversed(seq)) if reverse else seq

    rest_start = 2

    # ---- first block --------------------------------------------------------
    if use_reflect:
        # carry the header strip down to the square ...
        src = [M(0, i) for i in range(s)]
        ops.append(28)
        sels.append(sel_of(src))
        lay_row = (2 + s - 1) if seed_far else 2
        ops.append(30)
        sels.append(sel_of([origin_of([M(lay_row, i) for i in range(s)])]))
        # ... and give it a quarter turn, so it runs across the stripes instead
        # of along them.  Full s x s rectangle: exactly what Rotate must carry.
        ops.append(25)
        sels.append(sel_of(block_cells(2, s, s)))
        seed_col = 0 if seed_far else s - 1
        for j in range(s):
            rr, cc = M(2 + j, seed_col)
            # a seed whose colour is 0 never travelled; that cell still reads bgc
            cur[rr, cc] = ccols[j] if ccols[j] != 0 else bgc
        # each seed now spreads along its own stripe line
        for j in order(range(s)):
            paint(ccols[j], row_cells(2 + j))
        rest_start = 2 + s

    # ---- the repeats --------------------------------------------------------
    if use_tile:
        if not use_reflect:
            for j in order(range(w)):
                paint(stripe_color(2 + j), row_cells(2 + j))
            rest_start = 2 + w
        # the finished block, copied whole and stamped down the field
        ops.append(29)
        sels.append(sel_of(block_cells(2, w, w)))
        for t in order(range(1, nblocks)):
            r0 = 2 + t * w
            ops.append(30)
            sels.append(sel_of([origin_of(block_cells(r0, w, w))]))
            for r in range(r0, r0 + w):
                col = stripe_color(r)
                if col != 0:
                    for rr, cc in row_cells(r):
                        cur[rr, cc] = col
            # colour 0 does not travel with a Paste: lay those lines down here
            for r in range(r0, r0 + w):
                if stripe_color(r) == 0:
                    paint(0, row_cells(r))
        rest_start = 2 + nblocks * w

    # ---- whatever is left: one stripe line at a time ------------------------
    for r in order(range(rest_start, Hc)):
        paint(stripe_color(r), row_cells(r))

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])      # whole grid: an exact full rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bd4472b8"
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
                                f"for task bd4472b8"
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
                    f"Failed to build a complete episode for task bd4472b8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bd4472b8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
