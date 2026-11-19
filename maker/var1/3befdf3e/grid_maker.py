"""
ARC Task: 3befdf3e (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=0, ccols=None, **kwargs) -> dict:
    import random

    hlo = min(10, max_h)
    wlo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    if ccols is None:
        ccols = random.sample([c for c in range(10) if c != bgc], random.randint(2, 9))
    ccols = [c for c in ccols if c != bgc]
    if len(ccols) < 2:
        ccols = random.sample([c for c in range(10) if c != bgc], 2)

    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 40)))
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)

    # cycle the four framed-box shapes so one instance shows several of them
    shapecycle = [(1, 1), (1, 2), (2, 1), (2, 2)]
    random.shuffle(shapecycle)
    si = 0

    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        rh, rw = shapecycle[si % 4]
        si += 1
        fullh = (2 + 3 * rh)
        fullw = (2 + 3 * rw)
        cands = sfilter(inds, lambda ij: ij[0] <= h - fullh and ij[1] <= w - fullw)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        fullobj = backdrop(frozenset({loc, (loci + fullh - 1, locj + fullw - 1)}))
        if fullobj.issubset(inds):
            succ += 1
            inds = inds - fullobj
            incol, outcol = random.sample(ccols, 2)
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1),
                                          (loci + 2 * rh, locj + 2 * rw)}))
            ofoutcol = outbox(ofincol)
            gi = fill(gi, incol, ofincol)
            gi = fill(gi, outcol, ofoutcol)
            go = fill(go, outcol, ofincol)
            go = fill(go, incol, ofoutcol)
            ilocs = apply(first, ofoutcol)
            jlocs = apply(last, ofoutcol)
            ff = lambda ij: ij[0] in ilocs or ij[1] in jlocs
            addon = sfilter(fullobj - (ofincol | ofoutcol), ff)
            go = fill(go, outcol, addon)

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    hi, wi = A.shape

    # --- background: the colour the canvas is painted with, shared by every demo input
    cnt = Counter(A.flatten().tolist())
    try:
        if examples:
            for pair in examples:
                cnt.update(np.asarray(pair[0], dtype=int).flatten().tolist())
    except Exception:
        pass
    bgc = cnt.most_common(1)[0][0]

    # --- find every framed box (ring of outcol + inner block of incol)
    seen = np.zeros((hi, wi), dtype=bool)
    boxes = []
    for sr in range(hi):
        for sc in range(wi):
            if A[sr, sc] == bgc or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and A[nr, nc] != bgc:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            outcol = int(A[r0, c0])          # frame colour
            incol = int(A[r0 + 1, c0 + 1])   # inner-block colour
            rh = r1 - r0 - 1                 # inner height
            rw = c1 - c0 - 1                 # inner width

            ring = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)
                    if a in (r0, r1) or b in (c0, c1)]
            inner = [(a, b) for a in range(r0 + 1, r1) for b in range(c0 + 1, c1)]
            # arms: the frame band pushed outward by the inner size on all four sides
            arms_h = []
            for a in range(r0, r1 + 1):
                for b in range(max(0, c0 - rw), c0):
                    arms_h.append((a, b))
                for b in range(c1 + 1, min(wi, c1 + 1 + rw)):
                    arms_h.append((a, b))
            arms_v = []
            for b in range(c0, c1 + 1):
                for a in range(max(0, r0 - rh), r0):
                    arms_v.append((a, b))
                for a in range(r1 + 1, min(hi, r1 + 1 + rh)):
                    arms_v.append((a, b))

            boxes.append({'r0': r0, 'c0': c0, 'r1': r1, 'c1': c1,
                          'rh': rh, 'rw': rw, 'incol': incol, 'outcol': outcol,
                          'ring': ring, 'inner': inner,
                          'arms_h': arms_h, 'arms_v': arms_v})

    nb = len(boxes)
    csum = sum(b['incol'] + b['outcol'] for b in boxes)
    psum = sum(b['r0'] + b['c0'] for b in boxes)
    # route choices read off the input only -> same instance, same route
    order_key = (int(bgc) * 3 + hi + wi + nb) % 6
    style_key = (int(bgc) + csum + psum + nb) % 6

    cr_, cc_ = (hi - 1) / 2.0, (wi - 1) / 2.0
    if order_key == 0:                      # top-left first, sweeping down
        boxes.sort(key=lambda b: (b['r0'], b['c0']))
    elif order_key == 1:                    # bottom-right first, sweeping up
        boxes.sort(key=lambda b: (-b['r0'], -b['c0']))
    elif order_key == 2:                    # column by column, left to right
        boxes.sort(key=lambda b: (b['c0'], b['r0']))
    elif order_key == 3:                    # column by column, right to left
        boxes.sort(key=lambda b: (-b['c0'], -b['r0']))
    elif order_key == 4:                    # biggest box first
        boxes.sort(key=lambda b: (-(b['rh'] * b['rw']), b['r0'], b['c0']))
    else:                                   # from the middle of the grid outward
        boxes.sort(key=lambda b: ((b['r0'] - cr_) ** 2 + (b['c0'] - cc_) ** 2,
                                  b['r0'], b['c0']))

    G = A.copy()
    ops, sels = [], []

    def paint(color, cells):
        cells = [(r, c) for (r, c) in cells if 0 <= r < hi and 0 <= c < wi]
        if not cells:
            return
        if all(G[r, c] == color for (r, c) in cells):
            return                      # would change nothing -> not an operation
        ops.append(int(color))
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = color

    def flood(color, seed, expect):
        sr, sc = seed
        start = int(G[sr, sc])
        if start == color:
            return False
        reg = {(sr, sc)}
        st = [(sr, sc)]
        while st:
            ar, ac = st.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = ar + dr, ac + dc
                if 0 <= nr < hi and 0 <= nc < wi and (nr, nc) not in reg and G[nr, nc] == start:
                    reg.add((nr, nc))
                    st.append((nr, nc))
        if reg != set(expect):          # fill would leak past the intended region
            return False
        ops.append(10 + int(color))
        sels.append(sel_of([(sr, sc)]))
        for (r, c) in reg:
            G[r, c] = color
        return True

    for b in boxes:
        ring, inner = b['ring'], b['inner']
        arms_h, arms_v = b['arms_h'], b['arms_v']
        arms = arms_h + arms_v
        incol, outcol = b['incol'], b['outcol']
        r0, c0 = b['r0'], b['c0']

        if style_key == 0:
            # recolour the frame, then the middle together with its new arms
            paint(incol, ring)
            paint(outcol, inner + arms)
        elif style_key == 1:
            # grow the cross and fill the middle first, frame last
            paint(outcol, inner + arms)
            paint(incol, ring)
        elif style_key == 2:
            # frame, middle, then each pair of arms on its own axis
            paint(incol, ring)
            paint(outcol, inner)
            wide = (b['rw'] > b['rh']) or (b['rw'] == b['rh'] and (r0 + c0) % 2 == 0)
            if wide:
                paint(outcol, arms_h)
                paint(outcol, arms_v)
            else:
                paint(outcol, arms_v)
                paint(outcol, arms_h)
        elif style_key == 3:
            # flood the frame itself, then middle, then arms
            if not flood(incol, (r0, c0), ring):
                paint(incol, ring)
            paint(outcol, inner)
            paint(outcol, arms)
        elif style_key == 4:
            # flood the middle block, then frame, then arms
            if not flood(outcol, (r0 + 1, c0 + 1), inner):
                paint(outcol, inner)
            paint(incol, ring)
            paint(outcol, arms)
        else:
            # arms out first, then the middle, then the frame
            paint(outcol, arms)
            paint(outcol, inner)
            paint(incol, ring)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3befdf3e"
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
                                f"for task 3befdf3e"
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
                    f"Failed to build a complete episode for task 3befdf3e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3befdf3e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
