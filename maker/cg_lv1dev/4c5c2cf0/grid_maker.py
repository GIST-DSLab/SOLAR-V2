"""
ARC Task: 4c5c2cf0 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)
    rem = [c for c in rem if c != cc]
    objc = random.choice(rem)
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, cc=None, objc=None, **kwargs) -> dict:
    from random import randint, sample, choice

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if bgc is None:
        bgc = random.randrange(10)
    if cc is None:
        cc = random.choice([c for c in range(10) if c != bgc])
    if objc is None:
        objc = random.choice([c for c in range(10) if c not in (bgc, cc)])

    H = max(10, min(30, max_h))
    W = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, H))
    w = unifint(diff_lb, diff_ub, (10, W))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))

    sg = np.full((oh, ow), bgc, dtype=int)
    locc = (oh - 1, ow - 1)
    sg[locc] = cc
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def shp(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 5 and shp(cells) == (3, 3):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = sample(reminds, ncells)
    for (i, j) in cells:
        sg[i, j] = objc

    G1 = sg
    G2 = np.fliplr(sg)
    G3 = np.flipud(sg)
    G4 = np.fliplr(np.flipud(sg))
    vbar = np.full((oh, 1), bgc, dtype=int)
    hbar = np.full((1, ow), bgc, dtype=int)
    cp = np.full((1, 1), cc, dtype=int)
    topg = np.hstack([G1, vbar, G2])
    botg = np.hstack([G3, vbar, G4])
    ggm = np.hstack([hbar, cp, hbar])
    GG = np.vstack([topg, ggm, botg])

    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    go = np.full((h, w), bgc, dtype=int)
    go[loci:loci + GG.shape[0], locj:locj + GG.shape[1]] = GG
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = sg
    gi[go == cc] = cc

    k = choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_marker(I, bgc):
    # the 5-cell X: a colour whose cells are exactly the 3x3 box minus its 4 edge-neighbours
    for col in sorted(set(I.flatten().tolist())):
        if col == bgc:
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(I == col)]
        if len(pts) != 5:
            continue
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        if max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
            continue
        ci, cj = min(rs) + 1, min(cs) + 1
        want = {(ci, cj), (ci - 1, cj - 1), (ci - 1, cj + 1), (ci + 1, cj - 1), (ci + 1, cj + 1)}
        if set(pts) == want:
            return col, ci, cj
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the canvas colour (the pattern quadrant is a small fraction of the grid)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    marker = _find_marker(I, bgc)
    if marker is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    cc, ci, cj = marker

    # the pattern: every non-background cell that is not the X marker
    obj = [(int(r), int(c)) for r, c in np.argwhere((I != bgc) & (I != cc))]
    if not obj:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # Reading taken from the golf solution: every pattern cell is reflected about the
    # marker's centre - across the centre column, across the centre row, and through
    # the centre point. The X marker is its own mirror image, so it needs nothing.
    # Each mirror image is one unit: the cells that copy of the pattern lands on,
    # painted cell-by-cell in the colour each source cell carries.
    for kind in ("col", "row", "point"):
        by_colour = {}
        for (r, c) in obj:
            nr = r if kind == "col" else 2 * ci - r
            nc = c if kind == "row" else 2 * cj - c
            if 0 <= nr < hi and 0 <= nc < wi:
                by_colour.setdefault(int(I[r, c]), []).append((nr, nc))
        for colour, cells in sorted(by_colour.items()):
            ops.append(colour); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
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
                                f"for task 4c5c2cf0"
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
                    f"Failed to build a complete episode for task 4c5c2cf0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4c5c2cf0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
