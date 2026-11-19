"""
ARC Task: b775ac94 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    # Only the background colour has to stay the same across the episode. The rule
    # uses each marker's own colour as the copy's colour, so the other colours
    # (c1..c4) can change per object and per instance.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 5)
        ow = randint(2, 5)
        canv = canvas(bgc, (oh, ow))
        c1, c2, c3, c4 = sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = asindices(canv)
        for k in range(ncells):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        gLR = fill(canv, c1, obj)
        gLL = replace(vmirror(gLR), c1, c2)
        gUR = replace(hmirror(gLR), c1, c3)
        gUL = replace(vmirror(hmirror(gLR)), c1, c4)
        gU = hconcat(gUL, gUR)
        gL = hconcat(gLL, gLR)
        g = vconcat(gU, gL)
        g2 = canvas(bgc, (oh * 2, ow * 2))
        g2 = fill(g2, c1, shift(obj, (oh, ow)))
        nkeepcols = unifint(diff_lb, diff_ub, (1, 3))
        keepcols = sample((c2, c3, c4), nkeepcols)
        for cc in (c2, c3, c4):
            if cc not in keepcols:
                g = replace(g, cc, bgc)
            else:
                ofsi = -1 if cc in (c3, c4) else 0
                ofsj = -1 if cc in (c2, c4) else 0
                g2 = fill(g2, cc, {(oh + ofsi, ow + ofsj)})
        rotf = choice((identity, rot90, rot180, rot270))
        g = rotf(g)
        g2 = rotf(g2)
        obji = asobject(g2)
        objo = asobject(g)
        objo = sfilter(objo, lambda cij: cij[0] != bgc)
        obji = sfilter(obji, lambda cij: cij[0] != bgc)
        tonorm = invert(ulcorner(objo))
        obji = shift(obji, tonorm)
        objo = shift(objo, tonorm)
        oh, ow = shape(objo)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcdi = shift(obji, loc)
        plcdo = shift(objo, loc)
        plcdoi = toindices(plcdo)
        if plcdoi.issubset(inds):
            succ += 1
            inds = (inds - plcdoi) - mapply(neighbors, plcdoi)
            gi = paint(gi, plcdi)
            go = paint(go, plcdo)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    How the golf entry reads the task, and how that differs from the maker:
      The maker treats each object as a whole blob and draws one complete mirrored
      copy per marker with a single Color op.
      The golf entry starts from the 2x2 PIVOT: the blob's corner cell plus the
      marker pixels around it. It grows the blob outward from that corner, cell by
      cell (8-connected flood), and reflects every cell it reaches into all four
      quadrants around the pivot at the same time, giving each quadrant its
      marker's colour.
    The two readings agree on the output, the mirror axes (the pivot's lines are
    the blob's bounding-box edges) and the colours. What changes is the unit and
    the order. Here one Color op is one ring of the flood from the pivot, reflected
    into one marker's quadrant. Ring by ring, all of the marker quadrants grow
    outward from the pivot together.
    Ring 0 is the corner cell. Its reflections are the marker pixels themselves,
    which already hold the right colours, so no op is needed for it.
    Every selection comes from I alone: the component, the pivot and the reflected
    ring cells. O is never read.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the canvas colour; objects are small, sparse and spaced apart
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    N8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]

    ops, sels = [], []
    seen = set()
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or (sr, sc) in seen:
                continue
            # one placed object = one 8-connected non-background component
            comp = []
            dq = deque([(sr, sc)])
            seen.add((sr, sc))
            while dq:
                p = dq.popleft()
                comp.append(p)
                for a, b in N8:
                    q = (p[0] + a, p[1] + b)
                    if 0 <= q[0] < hi and 0 <= q[1] < wi and q not in seen \
                            and I[q] != bgc:
                        seen.add(q)
                        dq.append(q)

            groups = {}
            for p in comp:
                groups.setdefault(int(I[p]), []).append(p)
            # the blob is the multi-cell colour group; the markers are single pixels
            scol, S = max(groups.items(), key=lambda kv: (len(kv[1]), -kv[0]))
            if len(S) < 2:
                continue
            r0 = min(p[0] for p in S); r1 = max(p[0] for p in S)
            c0 = min(p[1] for p in S); c1 = max(p[1] for p in S)

            markers = []
            for mcol, cells in groups.items():
                if mcol == scol:
                    continue
                for mr, mc in cells:
                    dr = -1 if mr == r0 - 1 else (1 if mr == r1 + 1 else 0)
                    dc = -1 if mc == c0 - 1 else (1 if mc == c1 + 1 else 0)
                    if dr or dc:
                        markers.append((mr, mc, dr, dc, mcol))
            if not markers:
                continue

            # pivot: the blob's corner cell, i.e. the cell each marker is offset from
            mr, mc, dr0, dc0, _ = markers[0]
            corner = (mr - dr0, mc - dc0)
            cr, cc = corner

            # grow the blob outward from the pivot corner in BFS rings
            Sset = set(S)
            layers = [[corner]]
            vis = {corner}
            while True:
                nxt = []
                for p in layers[-1]:
                    for a, b in N8:
                        q = (p[0] + a, p[1] + b)
                        if q in Sset and q not in vis:
                            vis.add(q)
                            nxt.append(q)
                if not nxt:
                    break
                layers.append(sorted(nxt))

            markers.sort(key=lambda m: (m[2], m[3]))
            # ring 0 reflects onto the marker pixels themselves (already correct)
            for layer in layers[1:]:
                for _, _, dr, dc, mcol in markers:
                    # reflect across the pivot's horizontal and/or vertical line
                    cells = [(a if dr == 0 else 2 * cr + dr - a,
                              b if dc == 0 else 2 * cc + dc - b) for a, b in layer]
                    cells = [(a, b) for a, b in cells if 0 <= a < hi and 0 <= b < wi]
                    if cells:
                        ops.append(int(mcol))
                        sels.append(sel_of(sorted(set(cells))))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b775ac94"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b775ac94"
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
                                f"for task b775ac94"
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
                    f"Failed to build a complete episode for task b775ac94 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b775ac94-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
