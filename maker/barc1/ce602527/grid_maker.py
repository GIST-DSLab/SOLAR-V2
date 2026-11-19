"""
ARC Task: ce602527 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, c1, c2, c3 = random.sample(list(range(10)), 4)
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, c1=None, c2=None, c3=None, **kwargs) -> dict:
    if bgc is None or c1 is None or c2 is None or c3 is None:
        bgc, c1, c2, c3 = random.sample(list(range(10)), 4)
    h = unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    while True:
        objs = []
        for k in range(2):
            oh1 = unifint(diff_lb, diff_ub, (3, h // 3 - 1))
            ow1 = unifint(diff_lb, diff_ub, (3, w // 3 - 1))
            cc1 = canvas(bgc, (oh1, ow1))
            bounds1 = asindices(cc1)
            numcd1 = unifint(diff_lb, diff_ub, (0, (oh1 * ow1) // 2 - 4))
            numc1 = choice((numcd1, oh1 * ow1 - numcd1))
            numc1 = min(max(3, numc1), oh1 * ow1 - 3)
            obj1 = {choice(totuple(bounds1))}
            while len(obj1) < numc1 or shape(obj1) != (oh1, ow1):
                obj1.add(choice(totuple((bounds1 - obj1) & mapply(dneighbors, obj1))))
            objs.append(normalize(obj1))
        a, b = objs
        ag = fill(canvas(0, shape(a)), 1, a)
        bg = fill(canvas(0, shape(b)), 1, b)
        maxinh = min(height(a), height(b)) // 2 + 1
        maxinw = min(width(a), width(b)) // 2 + 1
        maxshp = (maxinh, maxinw)
        ag = crop(ag, (0, 0), maxshp)
        bg = crop(bg, (0, 0), maxshp)
        if ag != bg:
            break
    a, b = objs
    trgo = choice(objs)
    trgo2 = ofcolor(upscale(fill(canvas(0, shape(trgo)), 1, trgo), 2), 1)
    staysinh = unifint(diff_lb, diff_ub, (maxinh * 2, height(trgo) * 2))
    nout = height(trgo2) - staysinh
    loci = h - height(trgo2) + nout
    locj = randint(0, w - maxinw * 2)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c3, shift(trgo2, (loci, locj)))
    (lcol, lobj), (rcol, robj) = sample([(c1, a), (c2, b)], 2)
    cands = ofcolor(gi, bgc) - box(asindices(gi))
    lca = sfilter(cands, lambda ij: ij[1] < w // 3 * 2)
    lcands = sfilter(lca, lambda ij: shift(lobj, ij).issubset(lca))
    tries = 0
    while True:
        tries += 1
        if tries > 2000:
            raise RuntimeError("placement failed")
        lloc = choice(totuple(lcands))
        rloc = choice(totuple(lcands))
        lplcd = shift(lobj, lloc)
        rplcd = shift(robj, rloc)
        if lplcd.issubset(cands) and rplcd.issubset(cands) and len(lplcd & rplcd) == 0:
            break
    gi = fill(gi, lcol, shift(lobj, lloc))
    gi = fill(gi, rcol, shift(robj, rloc))
    go = fill(canvas(bgc, shape(trgo)), c1 if trgo == a else c2, trgo)
    mfs = (identity, rot90, rot180, rot270, cmirror, dmirror, hmirror, vmirror)
    mf = choice(mfs)
    gi, go = mf(gi), mf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    colors = [int(c) for c in np.unique(I)]

    # background: colour spanning the whole grid; fallback most common
    bgc = None
    for c in colors:
        rs, cs = np.where(I == c)
        if rs.min() == 0 and rs.max() == hi - 1 and cs.min() == 0 and cs.max() == wi - 1:
            bgc = c
            break
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    fgs = [c for c in colors if c != bgc]

    def bbox(col):
        rs, cs = np.where(I == col)
        return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())

    def mask(col):
        r0, c0, r1, c1 = bbox(col)
        return (I[r0:r1 + 1, c0:c1 + 1] == col).astype(int)

    def matches_scaled(small, other):
        # is `other` (possibly clipped by the border) a window of 2x-upscaled `small`?
        sm, om = mask(small), mask(other)
        if sm.sum() > om.sum():
            return False
        up = np.kron(sm, np.ones((2, 2), dtype=int))
        uh, uw = up.shape
        oh, ow = om.shape
        for r in range(uh - oh + 1):
            for cc in range(uw - ow + 1):
                if np.array_equal(up[r:r + oh, cc:cc + ow], om):
                    return True
        return False

    # the candidate is the object whose 2x-scaled shape shows up as another object
    target = None
    for c in fgs:
        if any(matches_scaled(c, o) for o in fgs if o != c):
            target = c
            break
    if target is None:
        target = fgs[0]

    r0, c0, r1, c1 = bbox(target)
    ho, wo = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # crop canvas to the target object's bounding rectangle (full rectangle intended)
    ops.append(33); sels.append([r0, c0, ho - 1, wo - 1])

    # any other object that intrudes into this rectangle becomes background, one object at a time
    for c in fgs:
        if c == target:
            continue
        cells = [(r - r0, cc - c0) for r in range(r0, r1 + 1) for cc in range(c0, c1 + 1)
                 if I[r, cc] == c]
        if cells:
            ops.append(bgc); sels.append(sel_of(cells))
    # if bgc is 0 it is transparent for the crop and already correct; nonzero bgc was copied over

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ce602527"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce602527"
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
                                f"for task ce602527"
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
                    f"Failed to build a complete episode for task ce602527 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce602527-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
