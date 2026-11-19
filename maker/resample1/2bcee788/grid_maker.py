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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "sepc": sepc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sepc, objc) -> dict:
    m = min(max_h, max_w, 30)
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
    gimini = shift(gimini, (loci, locj))
    gomini = shift(gomini, (loci, locj))
    gi = paint(fullg, gimini)
    go = paint(fullg, gomini)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background = colour whose cells span the largest bounding box (as in verifier)
    def bbox_area(col):
        rr, cc = np.where(I == col)
        return (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1), len(rr)
    colors = sorted(set(I.flatten().tolist()))
    bgc = max(colors, key=bbox_area)
    others = [c for c in colors if c != bgc]
    cnt = Counter(I.flatten().tolist())
    objc = max(others, key=lambda c: cnt[c])
    sepc = min(others, key=lambda c: cnt[c])
    obj = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    sep = [(int(r), int(c)) for r, c in zip(*np.where(I == sepc))]
    r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
    c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    s_rows = {r for r, _ in sep}; s_cols = {c for _, c in sep}

    # 1. background becomes 3
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == bgc))]
    ops.append(3); sels.append(sel_of(bg_cells))
    G = I.copy(); G[I == bgc] = 3

    horizontal = len(s_cols) == 1 and next(iter(s_cols)) not in range(c0, c1 + 1)
    if horizontal:
        s = next(iter(s_cols))
        if s > c1:
            dr, dc = r0, s
            mirror = lambda r, c: (r, 2 * s - 1 - c)
        else:
            dr, dc = r0, s - ow + 1
            mirror = lambda r, c: (r, 2 * s + 1 - c)
        flip_op = 26
    else:
        s = next(iter(s_rows))
        if s > r1:
            dr, dc = s, c0
            mirror = lambda r, c: (2 * s - 1 - r, c)
        else:
            dr, dc = s - oh + 1, c0
            mirror = lambda r, c: (2 * s + 1 - r, c)
        flip_op = 27

    if objc != 0:
        # duplicate the object's bounding rectangle (object + already-3 background,
        # all nonzero) onto the separator side, then mirror it in place
        ops.append(29); sels.append([r0, c0, oh - 1, ow - 1])   # full rectangle intended
        ops.append(30); sels.append([dr, dc, 0, 0])
        block = G[r0:r1 + 1, c0:c1 + 1].copy()
        G[dr:dr + oh, dc:dc + ow] = block
        flipped = np.fliplr(block) if flip_op == 26 else np.flipud(block)
        if not np.array_equal(flipped, block):
            ops.append(flip_op); sels.append([dr, dc, oh - 1, ow - 1])  # full rectangle
    else:
        # 0 can't be copied/pasted: paint the reflected object directly,
        # after clearing the separator marker
        ops.append(3); sels.append(sel_of(sep))
        ops.append(0); sels.append(sel_of([mirror(r, c) for r, c in obj]))

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
