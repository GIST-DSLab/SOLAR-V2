"""
ARC Task: 2bcee788 (RE-ARC) — LLM-generated grid_maker
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
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colour 3 is hard-wired (it is the output background), so it is never sampled.
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "sepc": sepc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sepc=None, objc=None) -> dict:
    if bgc is None or sepc is None or objc is None:
        cols = remove(3, interval(0, 10, 1))
        bgc, sepc, objc = sample(cols, 3)
    # The grid may be transposed later, so both sides must fit in min(max_h, max_w).
    m = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(20, m - 1))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(10, (m - 1) // 2))))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(diff_lb, diff_ub, (h + 1, max(h + 1, m)))
    fullw = unifint(diff_lb, diff_ub, (2 * w + 1, max(2 * w + 1, m)))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - h)
    locj = randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, shift(gimini, loc))
    go = paint(fullg, shift(gomini, loc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {"input": gi, "output": go}


def _side_from_cells(obj_rc, sep_rc):
    r0 = min(r for r, _ in obj_rc); r1 = max(r for r, _ in obj_rc)
    c0 = min(c for _, c in obj_rc); c1 = max(c for _, c in obj_rc)
    scols = {c for _, c in sep_rc}
    srows = {r for r, _ in sep_rc}
    if len(scols) == 1:
        sc = next(iter(scols))
        if sc > c1:
            return 'right'
        if sc < c0:
            return 'left'
    if len(srows) == 1:
        sr = next(iter(srows))
        if sr > r1:
            return 'bottom'
        if sr < r0:
            return 'top'
    # fallback: compare centroids
    mr = sum(r for r, _ in sep_rc) / len(sep_rc)
    mc = sum(c for _, c in sep_rc) / len(sep_rc)
    if mc > c1: return 'right'
    if mc < c0: return 'left'
    return 'bottom' if mr > r1 else 'top'


def derive_operations(I, O, examples=None):
    # The golf reading (march along the lines; at the marker line emit the
    # reversed lines seen so far; background -> 3) is the same rule as this one:
    # the marker line is where the reflection starts, the object is mirrored
    # across that edge, and the background turns 3. The golf code reflects the
    # whole strip of lines, but outside the object's bbox that strip is all
    # background, so the bbox is the same unit. Nothing in the route changes.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the colour whose bbox spans the whole picture
    best = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        key = (int((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)), int(len(rs)))
        if best is None or key > best[0]:
            best = (key, int(col))
    bgc = best[1]
    rest = [int(c) for c in np.unique(I) if int(c) != bgc]
    counts = {c: int((I == c).sum()) for c in rest}
    objc = max(rest, key=lambda c: counts[c])
    sepc = min(rest, key=lambda c: counts[c])

    obj_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    sep_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == sepc))]
    r0 = min(r for r, _ in obj_rc); r1 = max(r for r, _ in obj_rc)
    c0 = min(c for _, c in obj_rc); c1 = max(c for _, c in obj_rc)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    side = _side_from_cells(obj_rc, sep_rc)
    if side in ('right', 'left'):
        flip_op = 26
        dj = ow if side == 'right' else -ow
        dest_r, dest_c = r0, c0 + dj
        mirror_rc = [(r, c0 + c1 - c + dj) for r, c in obj_rc]
    else:
        flip_op = 27
        di = oh if side == 'bottom' else -oh
        dest_r, dest_c = r0 + di, c0
        mirror_rc = [(r0 + r1 - r + di, c) for r, c in obj_rc]

    # 1. the background becomes 3: all cells of the background colour
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == bgc))]
    if bgc != 3 and bg_cells:
        ops.append(3)
        sels.append(sel_of(bg_cells))

    if objc != 0:
        # 2. copy the object's whole bbox (object plus its 3-background).
        # The bbox is meant here: the whole rectangle, background included.
        ops.append(29)
        sels.append([r0, c0, oh - 1, ow - 1])
        # 3. paste it just past the marker line (this covers the marker)
        ops.append(30)
        sels.append([dest_r, dest_c, 0, 0])
        # 4. reflect the pasted rectangle in place
        src = np.where(I[r0:r1 + 1, c0:c1 + 1] == bgc, 3, I[r0:r1 + 1, c0:c1 + 1])
        mirrored = np.fliplr(src) if flip_op == 26 else np.flipud(src)
        if not np.array_equal(src, mirrored):
            ops.append(flip_op)
            sels.append([dest_r, dest_c, oh - 1, ow - 1])  # whole rectangle, on purpose
    else:
        # The object is colour 0, which Copy/Paste skip, so paint the reflected
        # object's cells directly. They cover the whole marker line.
        ops.append(0)
        sels.append(sel_of(mirror_rc))

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
                        f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
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
                                f"for task 2bcee788"
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
                    f"Failed to build a complete episode for task 2bcee788 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bcee788-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
