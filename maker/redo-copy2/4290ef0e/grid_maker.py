"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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


# The only discrete structural choice the generator makes: whether the centre
# cell of the figure carries its own extra-coloured dot.
VARIANTS = [{"has_center": True}, {"has_center": False}]


def sample_colors(num_examples=None) -> dict:
    # Only the background is a fixed role: the rule is about the SIZE of each
    # scattered frame, never about which colour it happens to be.
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


def generate(diff_lb, diff_ub, max_h, max_w, bgc, has_center=None) -> dict:
    if has_center is None:
        has_center = choice((True, False))
    cols = interval(0, 10, 1)
    dmax = min(7, max(2, max_h // 4), max(2, max_w // 4))
    if dmax < 2:
        dmax = 2
    while True:
        d = unifint(diff_lb, diff_ub, (2, dmax))
        h, w = d, d
        lo_h = min(max(4 * d, 2 * d + 2), max_h)
        lo_w = min(max(4 * d, 2 * d + 2), max_w)
        fullh = unifint(diff_lb, diff_ub, (lo_h, max_h))
        fullw = unifint(diff_lb, diff_ub, (lo_w, max_w))
        remcols = remove(bgc, cols)
        ccols = sample(remcols, d)
        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w - idx + 1)
            quad = fill(quad, c, (connect((idx, idx), (idx + linlen - 1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx + linlen - 1))))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if has_center:
            go = fill(go, choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            ow = width(obj)
            dh = max(0, ow // 2 - 1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - ow and ij[1] <= fullw - ow)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1
                loc = choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule, read off the INPUT alone:

      Each non-background colour in I is one scattered piece, and each piece is a
      square RING: the four corner-brackets of one square frame, so it is mirror
      symmetric about both of its own axes.  The pieces are the concentric rings
      of a single figure and nest by frame size -- widest frame outermost, each
      next-widest one cell further in, a lone pixel dead centre.

      A piece may have been dropped over the canvas edge, in which case part of it
      is missing; the surviving dimension of its bounding box still spans the whole
      frame, and its own mirror symmetry restores the cells the edge cut off.

    Route: take the widest ring's frame as the working frame, lay the background
    over it, redraw every ring inside it at the depth its own width dictates
    ((S - m) // 2 cells in from the border), then crop the canvas to that frame.

    Everything below -- the frame side S, where it sits, each ring's cells, each
    ring's depth -- is measured in I.  O is never read.
    """
    I = np.asarray(I, dtype=int)          # O is deliberately never read below
    hi, wi = I.shape

    # background: the canvas colour the pieces were scattered onto
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    pieces = []
    for col in sorted(set(int(v) for v in I.flatten()) - {bgc}):
        cells = [(r, c) for r in range(hi) for c in range(wi) if int(I[r, c]) == col]
        obs = set(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        m = max(bh, bw)                     # the ring's frame side

        def ring_from(fr, fc):
            """The whole ring implied by these cells sitting in a frame at (fr,fc):
            mirror them about both axes of that frame."""
            out = set()
            for r, c in cells:
                rr, cc = r - fr, c - fc
                if not (0 <= rr < m and 0 <= cc < m):
                    return None
                for a in (rr, m - 1 - rr):
                    for b in (cc, m - 1 - cc):
                        out.add((a, b))
            return out

        # A ring dropped over the canvas edge is short in one dimension, and what
        # is missing is off-grid: so its frame starts either at the near edge of
        # what survived or one frame-width back from the far edge.  Only one of
        # those stories has the whole ring meeting the canvas in exactly the cells
        # that are actually there.
        cand_r = [r0] if bh == m else [r0, r1 - m + 1]
        cand_c = [c0] if bw == m else [c0, c1 - m + 1]
        fr, fc, ring = r0, c0, None
        for cr in cand_r:
            for cc0 in cand_c:
                cand = ring_from(cr, cc0)
                if cand is None:
                    continue
                seen = {(cr + a, cc0 + b) for a, b in cand
                        if 0 <= cr + a < hi and 0 <= cc0 + b < wi}
                if seen == obs:
                    fr, fc, ring = cr, cc0, cand
                    break
            if ring is not None:
                break
        if ring is None:
            ring = ring_from(r0, c0) or {(r - r0, c - c0) for r, c in cells}

        pieces.append({"col": col, "m": m, "ring": sorted(ring), "fr": fr, "fc": fc})

    # widest frame first: that ring is the outermost one of the figure
    pieces.sort(key=lambda p: -p["m"])
    S = pieces[0]["m"]                       # the figure is S x S

    # the outer ring's own frame is the working frame (pulled inside the canvas
    # if that ring was the one hanging over the edge)
    R0 = max(0, min(pieces[0]["fr"], hi - S))
    C0 = max(0, min(pieces[0]["fc"], wi - S))

    ops, sels = [], []

    # 1. lay the background base over the whole frame square
    #    (bbox form: the selection really is that full rectangle)
    ops.append(bgc)
    sels.append([R0, C0, S - 1, S - 1])

    # 2. draw the rings into it, outermost first, each at its own depth
    for p in pieces:
        depth = (S - p["m"]) // 2
        cells = [(R0 + depth + r, C0 + depth + c) for r, c in p["ring"]]
        ops.append(int(p["col"]))
        sels.append(sel_of(cells))

    # 3. crop the canvas down to the assembled figure
    ops.append(33)
    sels.append([R0, C0, S - 1, S - 1])

    ops.append(34)
    sels.append([0, 0, S - 1, S - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
