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
import numpy as np
import random
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    remcols = [c for c in cols if c != bgc and c != fgc]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "fgc": fgc, "ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or fgc is None:
        bgc, fgc = sample(cols, 2)
    if ccols is None:
        remcols = remove(bgc, remove(fgc, cols))
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 8)))
    ccols = list(ccols)

    h = unifint(diff_lb, diff_ub, (min(8, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(8, max_w), max_w))

    # the object's bbox has to fit the canvas (odd side, 1/3/5)
    hcap = 2 if h >= 5 else (1 if h >= 3 else 0)
    wcap = 2 if w >= 5 else (1 if w >= 3 else 0)
    objh = unifint(diff_lb, diff_ub, (0, hcap))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else min(1, wcap), wcap))
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

    cellcount = len(obj)
    obj = {(choice(ccols), ij) for ij in obj}
    obj = normalize(obj)

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    loc = (loci, locj)
    plcd = shift(obj, loc)
    gi = paint(gi, plcd)
    go = paint(go, plcd)

    inds = (asindices(gi) - toindices(plcd)) - mapply(neighbors, toindices(plcd))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * cellcount))))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            go = paint(go, plcd)
            gi = fill(gi, fgc, {center(plcdi)})
            succ += 1
            inds = (inds - plcdi) - mapply(dneighbors, plcdi)
        tr += 1

    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """One multi-cell template plus a scatter of single marker cells; every
    marker becomes a copy of the template, centred on the marker.  So: copy the
    template once, then paste that one clipboard at each marker."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    full = [0, 0, ho - 1, wo - 1]              # exactly the full-grid rectangle
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 4-connected components of the non-background cells ---------------
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack, comp = [(r, c)], []
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] != bgc:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(comp)
    if not comps:
        return [34], [full]

    # the template is the one blob; every other component is a lone marker
    ti = max(range(len(comps)), key=lambda k: len(comps[k]))
    template = comps[ti]
    markers = [comps[k][0] for k in range(len(comps))
               if k != ti and len(comps[k]) == 1]
    if not markers:
        return [34], [full]

    rs = [r for r, _ in template]
    cs = [c for _, c in template]
    r0, c0 = min(rs), min(cs)
    oh, ow = max(rs) - r0 + 1, max(cs) - c0 + 1
    cr, cc = r0 + oh // 2, c0 + ow // 2        # the template's own centre cell
    body = [(r - cr, c - cc) for (r, c) in template]

    # the copies are laid down outward from the template they come from
    markers.sort(key=lambda m: (abs(m[0] - cr) + abs(m[1] - cc), m[0], m[1]))

    def inside(r, c):
        return 0 <= r < hi and 0 <= c < wi

    # what the rule asks for: the template's body centred on every marker
    T = I.copy()
    for (mr, mc) in markers:
        for (dr, dc) in body:
            if inside(mr + dr, mc + dc):
                T[mr + dr, mc + dc] = I[cr + dr, cc + dc]

    clip = I[r0:r0 + oh, c0:c0 + ow]           # CopyI drops cells that hold 0

    def replay(keep):
        """Replay the kept pastes; note which paste last wrote each cell."""
        G = I.copy()
        last = {}
        for k, (mr, mc) in enumerate(markers):
            if not keep[k]:
                continue
            for a in range(oh):
                for b in range(ow):
                    v = int(clip[a, b])
                    r, c = mr - oh // 2 + a, mc - ow // 2 + b
                    if v != 0 and inside(r, c):
                        G[r, c] = v
                        last[(r, c)] = k
        return G, last

    # A paste stays only if it ends up holding at least one cell right that the
    # input had wrong.  When the palette gave the whole body colour 0 the
    # clipboard carries nothing of it, so that copy is painted below instead.
    keep = [True] * len(markers)
    while True:
        G, last = replay(keep)
        useful = {k for (r, c), k in last.items()
                  if G[r, c] == T[r, c] and I[r, c] != T[r, c]}
        drop = [k for k in range(len(markers)) if keep[k] and k not in useful]
        if not drop:
            break
        for k in drop:
            keep[k] = False

    ops, sels = [], []
    if any(keep):
        # the full template rectangle, the background inside it included:
        # that whole region is what gets duplicated
        ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])
        for k, (mr, mc) in enumerate(markers):
            if keep[k]:
                ops.append(30)                 # paste origin = top-left corner
                sels.append([mr - oh // 2, mc - ow // 2, 0, 0])

    # Finish each copy where the clipboard could not: cells the palette coloured
    # 0 (Paste reads those as empty), and cells a neighbouring copy's rectangle
    # wrote background over on its way down.  One copy at a time, colour by colour.
    for (mr, mc) in markers:
        bycol = {}
        for (dr, dc) in body:
            r, c = mr + dr, mc + dc
            if inside(r, c) and G[r, c] != T[r, c]:
                bycol.setdefault(int(T[r, c]), []).append((r, c))
        for v in sorted(bycol):
            ops.append(v); sels.append(sel_of(bycol[v]))
            for (r, c) in bycol[v]:
                G[r, c] = v

    # and the original template itself, if a copy's rectangle reached into it
    bycol = {}
    for (r, c) in template:
        if G[r, c] != T[r, c]:
            bycol.setdefault(int(T[r, c]), []).append((r, c))
    for v in sorted(bycol):
        ops.append(v); sels.append(sel_of(bycol[v]))
        for (r, c) in bycol[v]:
            G[r, c] = v

    ops.append(34); sels.append(full)
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
