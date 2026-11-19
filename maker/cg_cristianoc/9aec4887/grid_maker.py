"""
ARC Task: 9aec4887 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def sample_colors(num_examples=None) -> dict:
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    H = max(12, min(30, max_h))
    W = max(12, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (12, H))
    w = _unifint(diff_lb, diff_ub, (12, W))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * ow for _ in range(oh)]
    for i in range(1, oh - 1):
        go[i][0] = c1
        go[i][ow - 1] = c2
    for j in range(1, ow - 1):
        go[0][j] = c3
        go[oh - 1][j] = c4
    bounds = {(i, j) for i in range(oh - 2) for j in range(ow - 2)}

    def nbrs(cells):
        out = set()
        for (i, j) in cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    objA = {random.choice(sorted(bounds))}
    ncells = _unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for _ in range(ncells - 1):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))

    def shp(o):
        rs = [i for i, _ in o]
        cs = [j for _, j in o]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(objA) != (oh - 2, ow - 2):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    boxcells = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in boxcells and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(sorted(rems))
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = go[i][j]
    for (i, j) in objA:
        gi[li + i][lj + j] = pc
    walls = [((i, 0), c1) for i in range(1, oh - 1)] + [((i, ow - 1), c2) for i in range(1, oh - 1)] \
        + [((0, j), c3) for j in range(1, ow - 1)] + [((oh - 1, j), c4) for j in range(1, ow - 1)]
    wallobjs = {}
    for (p, c) in walls:
        wallobjs.setdefault(c, []).append(p)
    for (i, j) in objA:
        a, b = i + 1, j + 1
        d = {c: min(abs(a - x) + abs(b - y) for x, y in ps) for c, ps in wallobjs.items()}
        m = min(d.values())
        win = [c for c, v in d.items() if v == m]
        go[a][b] = win[0] if len(win) == 1 else pc
    gi = np.array(gi)
    go = np.array(go)
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # partition non-background cells by colour: four straight walls + one blob
    by_col = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                by_col.setdefault(int(I[r, c]), []).append((r, c))
    horiz, vert, pc = [], [], None
    for col, cells in by_col.items():
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if len(rows) == 1:
            horiz.append((min(rows), col))
        elif len(cols) == 1:
            vert.append((min(cols), col))
        else:
            pc = col  # blob bbox is always >= 2x2, so it is never a straight line
    horiz.sort()
    vert.sort()
    (t, ctop), (b, cbot) = horiz
    (l, cleft), (rr, cright) = vert
    r0, c0 = t, l
    oh, ow = b - t + 1, rr - l + 1

    blob = sorted(by_col[pc])
    br = min(r for r, _ in blob)
    bc = min(c for _, c in blob)
    dr, dc = r0 + 1 - br, c0 + 1 - bc

    ops, sels = [], []

    # 1. carry the blob (one object) into the box interior:
    #    first Move grabs it, the rest continue the same grab with an empty selection.
    #    Its original footprint lies outside the box and is discarded by the crop below.
    if pc != 0:
        steps = [(20 if dr < 0 else 21)] * abs(dr) + [(23 if dc < 0 else 22)] * abs(dc)
        for k, op in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(blob if k == 0 else []))

    # 2. keep only the box — selection is exactly that full rectangle, background included
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    # blob cells in box-local coordinates (bbox top-left lands on (1,1))
    local = [(r - br + 1, c - bc + 1) for r, c in blob]
    if pc == 0:
        # a 0-coloured blob cannot be carried by Move (0 is "nothing"); stamp it instead
        ops.append(0)
        sels.append(sel_of(local))

    # 3. each wall claims the blob cells strictly nearest to it; ties keep the blob colour
    groups = {"top": [], "bottom": [], "left": [], "right": []}
    for (lr, lc) in local:
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        if len(win) == 1:
            groups[win[0]].append((lr, lc))
    for name, col in (("top", ctop), ("bottom", cbot), ("left", cleft), ("right", cright)):
        if groups[name]:
            ops.append(int(col))
            sels.append(sel_of(sorted(groups[name])))

    ops.append(34)
    sels.append([0, 0, oh - 1, ow - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9aec4887"
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
                                f"for task 9aec4887"
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
                    f"Failed to build a complete episode for task 9aec4887 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9aec4887-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
