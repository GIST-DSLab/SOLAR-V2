"""
ARC Task: e8dc4411 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, remc = sample(cols, 3)
    return {"bgc": bgc, "objc": objc, "remc": remc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None, remc=None) -> dict:
    h = unifint(diff_lb, diff_ub, (9, max(9, max_h)))
    w = unifint(diff_lb, diff_ub, (9, max(9, max_w)))
    d = unifint(diff_lb, diff_ub, (3, max(3, min(h, w) // 2 - 1)))
    c = canvas(bgc, (d, d))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= d // 2 and ij[1] >= d // 2)
    ncd = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 2)))
    nc = choice((ncd, len(inds) - ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = sample(totuple(inds), nc)
    cells = set(cells) | {choice(((d // 2, d // 2), (d // 2, d // 2 - 1)))}
    cells = cells | {(jj, ii) for ii, jj in cells}
    for k in range(4):
        c = fill(c, objc, cells)
        c = rot90(c)
    while palette(toobject(box(asindices(c)), c)) == frozenset({bgc}) and height(c) > 3:
        c = trim(c)
    obj = ofcolor(c, objc)
    od = height(obj)
    loci = randint(1, h - 2 * od)
    locj = randint(1, w - 2 * od)
    obj = shift(obj, (loci, locj))
    bd = backdrop(obj)
    p = 0
    while len(shift(obj, (p, p)) & bd) > 0:
        p += 1
    obj2 = shift(obj, (p, p))
    nbhs = mapply(neighbors, obj)
    while len(obj2 & nbhs) == 0:
        nbhs = mapply(neighbors, nbhs)
    indic = obj2 & nbhs
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    gi = fill(gi, remc, indic)
    go = tuple(e for e in gi)
    for k in range(30):
        newg = fill(go, remc, shift(obj, (p * (k + 1), p * (k + 1))))
        if newg == go:
            break
        go = newg
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background = canvas color the generator fills before placing objects (dominant)
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    fg = sorted([c for c in cnt if c != bgc], key=lambda c: -cnt[c])
    if len(fg) < 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    objc, remc = fg[0], fg[1]          # big shape vs. small indicator stub

    obj = [(r, c) for r in range(h) for c in range(w) if I[r, c] == objc]
    target = {(r, c) for r in range(h) for c in range(w) if O[r, c] == remc}

    def chain(dr, dc):
        """all in-bounds cells of the repeated diagonal copies"""
        out, k = set(), 1
        while True:
            cells = [(r + dr * k, c + dc * k) for r, c in obj]
            inb = [(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w]
            if not inb:
                break
            out |= set(inb)
            k += 1
            if k > 2 * max(h, w):
                break
        return out

    # find the diagonal step: copies march by (±p, ±p) until off-grid
    best = None
    for p in range(1, max(h, w) + 1):
        for sr in (1, -1):
            for sc in (1, -1):
                if chain(sr * p, sc * p) == target:
                    best = (sr * p, sc * p)
                    break
            if best:
                break
        if best:
            break

    if best is None:
        # fallback: paint the exact diff cells, grouped by colour
        by_col = {}
        for r in range(h):
            for c in range(w):
                if I[r, c] != O[r, c]:
                    by_col.setdefault(int(O[r, c]), []).append((r, c))
        for col, cells in by_col.items():
            ops.append(col); sels.append(sel_of(cells))
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    dr, dc = best
    k = 1
    while True:
        cells = [(r + dr * k, c + dc * k) for r, c in obj]
        inb = [(r, c) for r, c in cells if 0 <= r < h and 0 <= c < w]
        if not inb:
            break
        need = [(r, c) for r, c in inb if I[r, c] != remc]   # skip already-correct stub cells
        if need:
            ops.append(int(remc)); sels.append(sel_of(need))
        k += 1

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8dc4411"
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
                                f"for task e8dc4411"
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
                    f"Failed to build a complete episode for task e8dc4411 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8dc4411-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
