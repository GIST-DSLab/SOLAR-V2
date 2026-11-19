"""
ARC Task: 88a10436 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Episode-level palette.

    Roles:
      bgc   - canvas background
      fgc   - the colour of the single-cell markers (the 'stamp here' dots)
      ccols - the palette the multi-colour template is painted from.
              0 is deliberately excluded from ccols: the template is replicated
              with CopyI/Paste, and 0 is invisible to ARCLE's clipboard, so a
              0-coloured template cell could never be stamped.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    avail = [c for c in range(1, 10) if c != bgc and c != fgc]
    ncols = random.randint(1, min(8, len(avail)))
    ccols = random.sample(avail, ncols)
    return {"bgc": bgc, "fgc": fgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if fgc is None:
        fgc = choice([c for c in cols if c != bgc])
    if ccols is None:
        avail = [c for c in range(1, 10) if c != bgc and c != fgc]
        ccols = sample(avail, unifint(diff_lb, diff_ub, (1, len(avail))))
    ccols = list(ccols)

    h = unifint(diff_lb, diff_ub, (min(8, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(8, max_w), max_w))

    mh = max(0, min(2, (h - 1) // 2))
    mw = max(0, min(2, (w - 1) // 2))
    objh = unifint(diff_lb, diff_ub, (0, mh))
    lo = min(0 if objh > 0 else 1, mw)
    objw = unifint(diff_lb, diff_ub, (lo, mw))
    objh = objh * 2 + 1
    objw = objw * 2 + 1

    bb = asindices(canvas(-1, (objh, objw)))
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for k in range(ncells - 1):
        cands = totuple((bb - obj) & mapply(dneighbors, obj))
        if len(cands) == 0:
            break
        obj.add(choice(cands))
    while height(obj) != objh or width(obj) != objw:
        cands = totuple((bb - obj) & mapply(dneighbors, obj))
        if len(cands) == 0:
            break
        obj.add(choice(cands))
    ncells = len(obj)

    obj = {(choice(ccols), ij) for ij in obj}
    obj = normalize(obj)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = paint(go, plcd)

    # free cells: everything outside the template's *bounding box* (and its halo).
    # Working with whole bounding boxes (rather than just the object cells) keeps
    # every stamp's rectangle disjoint from every other stamp's rectangle, which
    # is what makes a rectangular CopyI/Paste of the template a faithful stamp.
    rect0 = frozenset((i, j) for i in range(loci, loci + objh)
                      for j in range(locj, locj + objw))
    inds = asindices(gi) - rect0 - mapply(neighbors, rect0)

    def _rect(a, b):
        return frozenset((i, j) for i in range(a, a + objh)
                         for j in range(b, b + objw))

    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * ncells))))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        a, b = choice(totuple(inds))
        rc = _rect(a, b)
        if rc.issubset(inds):
            go = paint(go, shift(obj, (a, b)))
            gi = fill(gi, fgc, {(a + objh // 2, b + objw // 2)})
            succ += 1
            inds = (inds - rc) - mapply(dneighbors, rc)
        tr += 1

    if succ == 0:
        for (a, b) in sorted(totuple(inds)):
            rc = _rect(a, b)
            if rc.issubset(inds):
                go = paint(go, shift(obj, (a, b)))
                gi = fill(gi, fgc, {(a + objh // 2, b + objw // 2)})
                break

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule: one multi-cell template sits somewhere on the canvas, and a scatter of
    single-cell markers says 'put a copy of the template here, centred on me'.

    Account: grab the template ONCE (CopyI over its bounding box), then Paste it
    at each marker, anchoring the copy so the template's centre lands on the
    marker.  One Paste per marker - one operation per repeat the rule makes.
    """
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- orthogonal connected components of the non-background cells ---
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                comp = []
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] \
                                and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(comp)

    ops, sels = [], []
    full = [(r, c) for r in range(h) for c in range(w)]

    if not comps:
        ops.append(34); sels.append(sel_of(full))
        return ops, sels

    # the template is the one big component; every marker is a lone cell
    ti = max(range(len(comps)), key=lambda k: len(comps[k]))
    tmpl = comps[ti]
    markers = sorted(comps[k][0] for k in range(len(comps))
                     if k != ti and len(comps[k]) == 1)

    if not markers:
        ops.append(34); sels.append(sel_of(full))
        return ops, sels

    r0 = min(r for r, _ in tmpl)
    c0 = min(c for _, c in tmpl)
    oh = max(r for r, _ in tmpl) - r0 + 1
    ow = max(c for _, c in tmpl) - c0 + 1

    # 1) grab the template. The selection is exactly the template's full bounding
    #    rectangle (background cells included) - that whole rectangle is the
    #    stamp, and the destinations are guaranteed clear rectangles.
    rect = [(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)]
    ops.append(28); sels.append(sel_of(rect))

    # 2) one Paste per marker, anchored so the template's centre covers the marker
    for (mr, mc) in markers:
        dr = mr - oh // 2
        dc = mc - ow // 2
        ops.append(30); sels.append(sel_of([(dr, dc)]))   # paste origin = top-left

    ops.append(34); sels.append(sel_of(full))
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
                        f"num_examples+1 ({num_examples + 1}) for task 88a10436"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 88a10436"
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
                                f"for task 88a10436"
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
                    f"Failed to build a complete episode for task 88a10436 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"88a10436-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
