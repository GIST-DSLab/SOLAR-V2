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
import random
from random import choice, sample
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        numcols = unifint(diff_lb, diff_ub, (2, 9))
        ccols = sample(remcols, numcols)
    ccols = tuple(ccols)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 40)))
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        rh = choice((1, 2))
        rw = choice((1, 2))
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
            incol, outcol = sample(ccols, 2)
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1), (loci + 2 * rh, locj + 2 * rw)}))
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

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour shared by I and every demonstration input
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    if bgc not in set(I.flatten().tolist()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # find framed boxes (4-connected non-bg components)
    seen = np.zeros((hi, wi), dtype=bool)
    boxes = []
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(p[0] for p in comp); r1 = max(p[0] for p in comp)
            c0 = min(p[1] for p in comp); c1 = max(p[1] for p in comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            boxes.append((r0, r1, c0, c1, int(I[r0, c0]), int(I[r0 + 1, c0 + 1])))

    ops, sels = [], []

    # Phase 1: per box, swap colours between frame and core.
    for (r0, r1, c0, c1, outcol, incol) in boxes:
        ring = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)
                if a in (r0, r1) or b in (c0, c1)]
        core = [(a, b) for a in range(r0 + 1, r1) for b in range(c0 + 1, c1)]
        ops.append(incol); sels.append(sel_of(ring))
        ops.append(outcol); sels.append(sel_of(core))

    # Phase 2: grow arms one side at a time (top, right, bottom, left), each side
    # applied to every box: a band beside the box, as wide as the box and as deep
    # as the core, painted in the original frame colour.
    for side in ("top", "right", "bottom", "left"):
        for (r0, r1, c0, c1, outcol, incol) in boxes:
            rh = r1 - r0 - 1
            rw = c1 - c0 - 1
            if side == "top":
                cells = [(a, b) for a in range(r0 - rh, r0) for b in range(c0, c1 + 1)]
            elif side == "bottom":
                cells = [(a, b) for a in range(r1 + 1, r1 + 1 + rh) for b in range(c0, c1 + 1)]
            elif side == "left":
                cells = [(a, b) for a in range(r0, r1 + 1) for b in range(c0 - rw, c0)]
            else:
                cells = [(a, b) for a in range(r0, r1 + 1) for b in range(c1 + 1, c1 + 1 + rw)]
            cells = [(a, b) for a, b in cells if 0 <= a < hi and 0 <= b < wi]
            if cells:
                ops.append(outcol); sels.append(sel_of(cells))

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
