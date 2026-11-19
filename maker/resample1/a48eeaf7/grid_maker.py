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
    # dot colour must be non-zero: ARCLE Move cannot carry 0-valued cells
    dotc = random.randint(1, 9)
    bgc, sqc = random.sample([c for c in range(10) if c != dotc], 2)
    return {"bgc": bgc, "sqc": sqc, "dotc": dotc}

def _unif(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    d = random.uniform(diff_lb, diff_ub)
    return max(lb, min(ub, lb + int(round((ub - lb) * d))))

def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc, dotc) -> dict:
    k = random.choice([0, 1, 2, 3])
    HH, WW = (max_h, max_w) if k % 2 == 0 else (max_w, max_h)
    h = _unif(diff_lb, diff_ub, 8, max(8, min(30, HH)))
    w = _unif(diff_lb, diff_ub, 8, max(8, min(30, WW)))
    ih = _unif(diff_lb, diff_ub, 2, h // 2)
    iw = _unif(diff_lb, diff_ub, 2, w // 2)
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci:loci+ih, locj:locj+iw] = sqc
    go[loci:loci+ih, locj:locj+iw] = sqc
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
    num = _unif(diff_lb, diff_ub, 1, len(srarr))
    locs = set(random.sample(range(len(srarr)), num))
    for j, (s, d) in enumerate(zip(srarr, dearr)):
        if j in locs:
            gi[s] = dotc
            go[d] = dotc
    ncorn = _unif(diff_lb, diff_ub, 0, 4)
    corners = [((loci - 1, locj - 1), (-1, -1)), ((loci - 1, locj + iw), (-1, 1)),
               ((loci + ih, locj - 1), (1, -1)), ((loci + ih, locj + iw), (1, 1))]
    for (cr, cc), (dr, dc) in corners[:ncorn]:
        go[cr, cc] = dotc
        cands = []
        r, c = cr + dr, cc + dc
        while 0 <= r < h and 0 <= c < w:
            cands.append((r, c)); r += dr; c += dc
        gi[random.choice(cands)] = dotc
    gi = np.rot90(gi, k).copy()
    go = np.rot90(go, k).copy()
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # rectangle = largest single-colour set that exactly fills its bounding box
    rect = None
    for col in set(I.flatten().tolist()) - {bgc}:
        rs, cs = np.where(I == col)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        if len(rs) == (r1 - r0 + 1) * (c1 - c0 + 1):
            if rect is None or len(rs) > rect[0]:
                rect = (len(rs), col, r0, r1, c0, c1)
    _, sqc, r0, r1, c0, c1 = rect
    outbox = [(r, c) for r in range(r0 - 1, r1 + 2) for c in range(c0 - 1, c1 + 2)
              if (r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1))]
    dots = [(r, c) for r in range(hi) for c in range(wi)
            if I[r, c] != bgc and I[r, c] != sqc]

    def target(p):
        return min(outbox, key=lambda q: abs(q[0]-p[0]) + abs(q[1]-p[1]))
    # nearer dots first
    dots.sort(key=lambda p: abs(target(p)[0]-p[0]) + abs(target(p)[1]-p[1]))
    ops, sels = [], []
    for (r, c) in dots:
        tr, tc = target((r, c))
        dr, dc = tr - r, tc - c
        if dr == 0 and dc == 0:
            continue
        vs = [(20 if dr < 0 else 21)] * abs(dr)
        hs = [(23 if dc < 0 else 22)] * abs(dc)
        seq = []
        # interleave vertical/horizontal steps for diagonal (corner) glides
        for i in range(max(len(vs), len(hs))):
            if i < len(vs): seq.append(vs[i])
            if i < len(hs): seq.append(hs[i])
        for i, op in enumerate(seq):
            ops.append(op)
            # first step grabs the dot; later steps continue the same grab
            sels.append(sel_of([(r, c)]) if i == 0 else sel_of([]))
        if bgc != 0:
            # only the original footprint is left as 0 by the slide
            ops.append(bgc); sels.append(sel_of([(r, c)]))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
