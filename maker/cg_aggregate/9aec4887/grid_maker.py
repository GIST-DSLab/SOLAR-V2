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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule only depends on where the walls are and where the blob is, not on
    # which colours they use. The generator still draws six distinct colours, so
    # all six are fixed once per episode.
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int(round((b - a) * diff_lb))
    hi = a + int(round((b - a) * diff_ub))
    lo, hi = min(lo, hi), max(lo, hi)
    return random.randint(lo, hi)


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, pc=None,
             c1=None, c2=None, c3=None, c4=None, **kwargs) -> dict:
    if None in (bgc, pc, c1, c2, c3, c4):
        bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * ow for _ in range(oh)]
    for r in range(1, oh - 1):
        go[r][0] = c1
        go[r][ow - 1] = c2
    for c in range(1, ow - 1):
        go[0][c] = c3
        go[oh - 1][c] = c4
    # grow the blob with 8-neighbourhood steps inside (oh-2)x(ow-2)
    bh, bw = oh - 2, ow - 2
    bounds = {(i, j) for i in range(bh) for j in range(bw)}
    objA = {random.choice(sorted(bounds))}

    def frontier():
        f = set()
        for (i, j) in objA:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if (di or dj) and (i + di, j + dj) in bounds and (i + di, j + dj) not in objA:
                        f.add((i + di, j + dj))
        return sorted(f)

    ncells = _unifint(diff_lb, diff_ub, (1, (bh * bw) // 2))
    for _ in range(ncells - 1):
        objA.add(random.choice(frontier()))

    def shp():
        rs = [i for i, _ in objA]
        cs = [j for _, j in objA]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp() != (bh, bw):
        objA.add(random.choice(frontier()))
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    boxcells = {(loci + i, locj + j) for i in range(oh) for j in range(ow)}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in boxcells and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = go[i][j]
    for (i, j) in objA:
        if 0 <= li + i < h and 0 <= lj + j < w:
            gi[li + i][lj + j] = pc
    wallcol = {"top": c3, "bottom": c4, "left": c1, "right": c2}
    for (i, j) in objA:
        lr, lc = i + 1, j + 1
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        go[lr][lc] = wallcol[win[0]] if len(win) == 1 else pc
    k = random.randint(0, 3)
    for _ in range(k):
        gi = _rot90(gi)
        go = _rot90(go)
    return {"input": gi, "output": go}


def _parse(I, bgc):
    H, W = len(I), len(I[0])
    cells = {}
    for r in range(H):
        for c in range(W):
            v = I[r][c]
            if v != bgc:
                cells.setdefault(v, []).append((r, c))
    horiz, vert, other = [], [], []
    for col, cl in cells.items():
        rs = {r for r, _ in cl}
        cs = {c for _, c in cl}
        if len(rs) == 1 and len(cl) >= 2 and max(cs) - min(cs) + 1 == len(cl):
            horiz.append((col, cl))
        elif len(cs) == 1 and len(cl) >= 2 and max(rs) - min(rs) + 1 == len(cl):
            vert.append((col, cl))
        else:
            other.append((col, cl))
    if len(horiz) != 2 or len(vert) != 2 or len(other) != 1:
        return None
    horiz.sort(key=lambda t: t[1][0][0])
    vert.sort(key=lambda t: t[1][0][1])
    top, bot = horiz
    left, right = vert
    r0 = top[1][0][0]
    r1 = bot[1][0][0]
    c0 = left[1][0][1]
    c1 = right[1][0][1]
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    if {c for _, c in top[1]} != set(range(c0 + 1, c1)):
        return None
    if {c for _, c in bot[1]} != set(range(c0 + 1, c1)):
        return None
    if {r for r, _ in left[1]} != set(range(r0 + 1, r1)):
        return None
    if {r for r, _ in right[1]} != set(range(r0 + 1, r1)):
        return None
    pc, blob = other[0]
    br = min(r for r, _ in blob)
    bc = min(c for _, c in blob)
    return {"r0": r0, "c0": c0, "oh": oh, "ow": ow, "pc": pc, "blob": blob,
            "br": br, "bc": bc, "ctop": top[0], "cbot": bot[0],
            "cleft": left[0], "cright": right[0]}


def derive_operations(I, O=None, examples=None):
    I = [list(map(int, row)) for row in I]
    info = None
    for bgc, _ in Counter(v for row in I for v in row).most_common():
        info = _parse(I, bgc)
        if info is not None:
            break

    r0, c0, oh, ow = info["r0"], info["c0"], info["oh"], info["ow"]
    pc = info["pc"]
    blob = sorted(info["blob"])
    dr = (r0 + 1) - info["br"]
    dc = (c0 + 1) - info["bc"]

    ops, sels = [], []

    # 1. carry the blob into the box: its top-left lands on interior cell (1,1).
    #    The blob is a translated object: grab it once, then use empty moves.
    if pc != 0:
        steps = []
        steps += [20 if dr < 0 else 21] * abs(dr)
        steps += [23 if dc < 0 else 22] * abs(dc)
        for k, op in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(blob) if k == 0 else sel_of([]))
        # The vacated footprint lies outside the box and is cropped away next,
        # so it gets no repair.
    else:
        # A colour-0 object cannot be carried by Move (0 is "nothing" to ARCLE),
        # so stamp its translated footprint into the interior instead.
        ops.append(0)
        sels.append(sel_of([(r + dr, c + dc) for r, c in blob]))

    # 2. the answer is the box alone: crop to it (full rectangle, background included)
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])

    # 3. the box splits into the four regions nearest each wall (triangles /
    #    trapezoids cut by the corner diagonals). Each wall recolours the blob
    #    cells in its region; cells on a diagonal tie keep the blob colour.
    local = [(r - info["br"] + 1, c - info["bc"] + 1) for r, c in blob]
    walls = [("top", info["ctop"]), ("bottom", info["cbot"]),
             ("left", info["cleft"]), ("right", info["cright"])]
    claim = {}
    for (lr, lc) in local:
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        claim[(lr, lc)] = win[0] if len(win) == 1 else None
    for name, col in walls:
        grp = sorted(p for p in local if claim[p] == name)
        if grp and col != pc:
            ops.append(int(col))
            sels.append(sel_of(grp))

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
