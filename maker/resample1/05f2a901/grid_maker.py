"""
ARC Task: 05f2a901 (RE-ARC) — LLM-generated grid_maker
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
try:
    from dsl import *
    from utils import unifint
except Exception:
    pass
from random import choice, randint, sample


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # moving object must be non-zero: ARCLE Move only carries non-zero cells
    fgc = random.choice([c for c in cols if c not in (bgc, 0)])
    destc = random.choice([c for c in cols if c not in (bgc, fgc)])
    return {"bgc": bgc, "fgc": fgc, "destc": destc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, destc=2) -> dict:
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    objh = unifint(diff_lb, diff_ub, (2, min(w // 2, h // 2)))
    objw = unifint(diff_lb, diff_ub, (objh, w // 2))
    bb = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loci = unifint(diff_lb, diff_ub, (3, h - objh))
    locj = unifint(diff_lb, diff_ub, (0, w - objw))
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(obj, loc)
    gi = fill(gi, fgc, obj)
    sqd = randint(1, min(w, loci - 1))
    locisq = randint(0, loci - sqd - 1)
    locjsq = randint(locj - sqd + 1, locj + objw - 1)
    sq = backdrop(frozenset({(locisq, locjsq), (locisq + sqd - 1, locjsq + sqd - 1)}))
    gi = fill(gi, destc, sq)
    go = fill(go, destc, sq)
    while len(obj & sq) == 0:
        obj = shift(obj, (-1, 0))
    obj = shift(obj, (1, 0))
    go = fill(go, fgc, obj)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = sorted(set(I.flatten().tolist()))

    def is_rect(c):
        rs, cs = np.where(I == c)
        return len(rs) == (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)

    def is_shift(a, b):
        if a.shape != b.shape:
            return False
        pa, pb = np.argwhere(a), np.argwhere(b)
        if len(pa) != len(pb) or len(pa) == 0:
            return False
        d = pb.min(0) - pa.min(0)
        if not d.any():
            return False
        return set(map(tuple, pa + d)) == set(map(tuple, pb))

    # Roles read from the demonstrations: the colour whose mask slides is the
    # moving object; the fixed square-shaped full rectangle is the anchor.
    fgc = destc = None
    if examples:
        for ex_i, ex_o in examples:
            ex_i = np.asarray(ex_i, dtype=int)
            ex_o = np.asarray(ex_o, dtype=int)
            for c in set(ex_i.flatten().tolist()):
                if is_shift(ex_i == c, ex_o == c):
                    fgc = c
            if fgc is not None:
                for c in [c for c in set(ex_i.flatten().tolist()) if c != fgc]:
                    rs, cs = np.where(ex_i == c)
                    if len(rs) == (rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1) and \
                       (rs.max() - rs.min()) == (cs.max() - cs.min()):
                        destc = c
                break
    if fgc not in colors or destc not in colors or fgc == destc:
        rects = [c for c in colors if is_rect(c)]
        destc = rects[0] if len(rects) == 1 else destc
        border = np.concatenate([I[0], I[-1], I[:, 0], I[:, -1]]).tolist()
        others = [c for c in colors if c != destc]
        bgc = max(others, key=lambda c: (border.count(c), (I == c).sum()))
        fgc = [c for c in others if c != bgc][0]
    bgc = [c for c in colors if c not in (fgc, destc)][0]

    obj = set(map(tuple, np.argwhere(I == fgc)))
    sq = set(map(tuple, np.argwhere(I == destc)))
    orr = [r for r, _ in obj]; occ = [c for _, c in obj]
    srr = [r for r, _ in sq]; scc = [c for _, c in sq]
    # gravitate: columns overlap -> vertical slide, else horizontal
    if max(occ) >= min(scc) and max(scc) >= min(occ):
        dr, dc = (1, 0) if min(srr) > max(orr) else (-1, 0)
    else:
        dr, dc = (0, 1) if min(scc) > max(occ) else (0, -1)
    op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]

    # slide until one more step would collide with the square
    steps = 0
    cur = set(obj)
    while True:
        nxt = {(r + dr, c + dc) for r, c in cur}
        if nxt & sq:
            break
        cur = nxt
        steps += 1

    ops, sels = [], []
    if fgc != 0:
        cur = list(obj)
        for k in range(steps):
            ops.append(op)
            sels.append(sel_of(cur) if k == 0 else sel_of([]))  # grab once, then continue
            cur = [(r + dr, c + dc) for r, c in cur]
        hole = sorted(obj - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))  # repair original footprint only
    else:
        # colour 0 cannot be carried by ARCLE Move: erase and repaint
        ops.append(bgc); sels.append(sel_of(obj))
        ops.append(0); sels.append(sel_of(cur))
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
                        f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05f2a901"
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
                                f"for task 05f2a901"
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
                    f"Failed to build a complete episode for task 05f2a901 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05f2a901-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
