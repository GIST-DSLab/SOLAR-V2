"""
ARC Task: 72ca375d (RE-ARC) — LLM-generated grid_maker
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
from random import randint, choice
import numpy as np
from collections import Counter
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    trgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "trgc": trgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, trgc=1, **kw) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = unifint(diff_lb, diff_ub, (2, 8))
    srcobjwh = unifint(diff_lb, diff_ub, (1, 4))
    bnds = asindices(canvas(-1, (srcobjh, srcobjwh)))
    spi = randint(0, srcobjh - 1)
    sp = (spi, srcobjwh - 1)
    srcobj = {sp}
    bnds = remove(sp, bnds)
    ncellsd = unifint(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = unifint(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for k in range(ncells - 1):
        srcobj.add(choice(totuple((bnds - srcobj) & mapply(neighbors, srcobj))))
    srcobj = normalize(srcobj)
    srcobj = srcobj | shift(vmirror(srcobj), (0, width(srcobj)))
    srcobjh, srcobjw = shape(srcobj)
    remcols = remove(bgc, cols)
    go = canvas(bgc, (srcobjh, srcobjw))
    go = fill(go, trgc, srcobj)
    loci = randint(0, h - srcobjh)
    locj = randint(0, w - srcobjw)
    locc = (loci, locj)
    gi = canvas(bgc, (h, w))
    shftd = shift(srcobj, locc)
    gi = fill(gi, trgc, shftd)
    indss = asindices(gi)
    indss = (indss - shftd) - mapply(neighbors, shftd)
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    remcands = asindices(canvas(-1, (8, 8))) - srcobj
    while succ < nobjs and tr <= maxtrials:
        if len(indss) == 0:
            break
        while True:
            newobj = {e for e in srcobj}
            numperti = unifint(diff_lb, diff_ub, (1, 63))
            numpert = 64 - numperti
            for _ in range(numpert):
                isadd = choice((True, False))
                if isadd and len(newobj) < 64:
                    cndds = totuple((remcands - newobj) & mapply(neighbors, newobj))
                    if len(cndds) == 0:
                        break
                    newobj.add(choice(cndds))
                if not isadd and len(newobj) > 2:
                    newobj = remove(choice(totuple(newobj)), newobj)
            newobj = normalize(newobj)
            a, b = shape(newobj)
            cc = canvas(-1, (a + 2, b + 2))
            cc2 = compress(fill(cc, -2, shift(newobj, (1, 1))))
            newobj = toindices(argmax(colorfilter(objects(cc2, T, T, F), -2), size))
            if newobj != vmirror(newobj):
                break
        col = choice(remcols)
        loccands = sfilter(indss, lambda ij: shift(newobj, ij).issubset(indss))
        if len(loccands) == 0:
            tr += 1
            continue
        locc = choice(totuple(loccands))
        newobj = shift(newobj, locc)
        gi = fill(gi, col, newobj)
        succ += 1
        indss = (indss - newobj) - mapply(neighbors, newobj)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour shared by every input of the episode
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    def bbox(cells):
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def comps_of(mask):
        seen = np.zeros_like(mask, dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if not mask[r, c] or seen[r, c]:
                    continue
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    cr, cc = st.pop()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                st.append((nr, nc))
                out.append(cells)
        return out

    # Reading 1 (colour as the unit): for each colour, the rectangle spanned by
    # all its cells; the target colour is the one whose rectangle of I reads the
    # same left-to-right as right-to-left.
    cands = []
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = list(zip(*np.nonzero(I == col)))
        r0, c0, bh, bw = bbox(cells)
        sub = I[r0:r0 + bh, c0:c0 + bw]
        if np.array_equal(sub, sub[:, ::-1]):
            ncomp = len(comps_of(I == col))
            cands.append(((ncomp == 1, len(cells)), (r0, c0, bh, bw), col))
    best, tcol = (max(cands)[1], max(cands)[2]) if cands else (None, None)

    # Reading 2 (fallback when the target colour is shared by a distractor):
    # individual 8-connected same-colour objects, symmetric under vmirror.
    if best is None:
        bk = None
        for col in sorted(set(I.flatten().tolist()) - {bgc}):
            for cells in comps_of(I == col):
                r0, c0, bh, bw = bbox(cells)
                m = np.zeros((bh, bw), dtype=bool)
                for (r, c) in cells:
                    m[r - r0, c - c0] = True
                if np.array_equal(m, m[:, ::-1]):
                    k = (len(cells), bh * bw)
                    if bk is None or k > bk[0]:
                        bk = (k, (r0, c0, bh, bw), col)
        best, tcol = (bk[1], bk[2]) if bk else ((0, 0, hi, wi), None)

    r0, c0, bh, bw = best
    ops, sels = [], []
    if (r0, c0, bh, bw) != (0, 0, hi, wi):
        # crop the canvas to the full rectangle around the symmetric object
        # (whole rectangle, background included)
        ops.append(33)
        sels.append([r0, c0, bh - 1, bw - 1])
    # a distractor of another colour poking into the rectangle is not part of
    # the symmetric object: paint each such colour's cells back to background
    if tcol is not None:
        sub = I[r0:r0 + bh, c0:c0 + bw]
        for col in sorted(set(sub.flatten().tolist()) - {bgc, tcol}):
            cells = [(r, c) for r in range(bh) for c in range(bw) if sub[r, c] == col]
            ops.append(bgc)
            sels.append(sel_of(cells))
    ops.append(34)
    sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72ca375d"
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
                                f"for task 72ca375d"
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
                    f"Failed to build a complete episode for task 72ca375d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72ca375d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
