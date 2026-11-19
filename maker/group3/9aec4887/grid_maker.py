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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # the blob is carried into the box with Move ops; ARCLE only grabs nonzero
    # cells, so the blob colour is drawn from the nonzero colours
    pc = random.choice(range(1, 10))
    rest = [c for c in cols if c != pc]
    bgc, c1, c2, c3, c4 = random.sample(rest, 5)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    h = _unifint(diff_lb, diff_ub, (12, max(12, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (12, max(12, min(30, max_w))))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((oh, ow), bgc, dtype=int)
    go[1:oh - 1, 0] = c1
    go[1:oh - 1, ow - 1] = c2
    go[0, 1:ow - 1] = c3
    go[oh - 1, 1:ow - 1] = c4
    bh, bw = oh - 2, ow - 2
    bounds = {(i, j) for i in range(bh) for j in range(bw)}
    objA = {random.choice(sorted(bounds))}

    def nbrs(cells):
        out = set()
        for (i, j) in cells:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    def shp(o):
        rs = [i for i, _ in o]
        cs = [j for _, j in o]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    ncells = _unifint(diff_lb, diff_ub, (1, (bh * bw) // 2))
    for _ in range(ncells - 1):
        cand = sorted((bounds - objA) & nbrs(objA))
        objA.add(random.choice(cand))
    while shp(objA) != (bh, bw):
        cand = sorted((bounds - objA) & nbrs(objA))
        objA.add(random.choice(cand))

    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    plcdi = {(i + loci, j + locj) for i in range(oh) for j in range(ow)
             if not ((i in (0, oh - 1)) and (j in (0, ow - 1)))}
    full = {(i, j) for i in range(h) for j in range(w)}
    rems = sorted(ij for ij in full - plcdi
                  if loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = random.choice(rems)
    gi[loci:loci + oh, locj:locj + ow] = go
    for (i, j) in objA:
        gi[i + loc[0], j + loc[1]] = pc
    for (i, j) in objA:
        lr, lc = i + 1, j + 1
        d = {c3: lr, c4: oh - 1 - lr, c1: lc, c2: ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        go[lr, lc] = win[0] if len(win) == 1 else pc
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _parse(I):
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    hlines, vlines, blob = [], [], None
    for col in set(I.flatten().tolist()) - {bgc}:
        cells = [tuple(int(v) for v in x) for x in np.argwhere(I == col)]
        rs = {r for r, _ in cells}
        cs = {c for _, c in cells}
        if len(rs) == 1 and len(cs) > 1:
            hlines.append((min(rs), min(cs), max(cs), col))
        elif len(cs) == 1 and len(rs) > 1:
            vlines.append((min(cs), min(rs), max(rs), col))
        else:
            blob = (col, cells)
    hlines.sort()
    vlines.sort()
    top, bot = hlines[0], hlines[1]
    left, right = vlines[0], vlines[1]
    r0, r1 = top[0], bot[0]
    c0, c1 = left[0], right[0]
    return dict(bgc=bgc, r0=r0, c0=c0, oh=r1 - r0 + 1, ow=c1 - c0 + 1,
                ctop=top[3], cbot=bot[3], cleft=left[3], cright=right[3],
                pc=blob[0], blob=blob[1])


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    info = _parse(I)
    r0, c0, oh, ow = info["r0"], info["c0"], info["oh"], info["ow"]
    blob = info["blob"]
    br = min(r for r, _ in blob)
    bc = min(c for _, c in blob)

    # 1. the blob has exactly the box-interior footprint: slide it into the box
    #    (one grab, then empty-selection continuation moves)
    dr = (r0 + 1) - br
    dc = (c0 + 1) - bc
    steps = []
    steps += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
    steps += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)

    ops, sels = [], []
    cur = list(blob)
    first = True
    for op, (sr, sc) in steps:
        ops.append(op)
        sels.append(sel_of(cur) if first else sel_of([]))
        first = False
        cur = [(r + sr, c + sc) for r, c in cur]
    # the vacated footprint lies outside the box, which the final crop discards

    # 2. each wall recolours the moved blob cells strictly nearest to it;
    #    cells tied between walls keep the blob colour
    walls = [("top", info["ctop"]), ("bottom", info["cbot"]),
             ("left", info["cleft"]), ("right", info["cright"])]
    groups = {k: [] for k, _ in walls}
    for (r, c) in sorted(cur):
        lr, lc = r - r0, c - c0
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        if len(win) == 1:
            groups[win[0]].append((r, c))
    for name, col in walls:
        if groups[name]:
            ops.append(int(col))
            sels.append(sel_of(groups[name]))

    # 3. crop to the box — the selection is exactly that full rectangle
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])
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
