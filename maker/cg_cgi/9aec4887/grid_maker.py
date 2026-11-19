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
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))

    def nbrs(cell):
        r, c = cell
        return {(r + i, c + j) for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)}

    H = max(12, min(30, max_h))
    W = max(12, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (12, H))
    w = unifint(diff_lb, diff_ub, (12, W))
    oh = unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((oh, ow), bgc, dtype=int)
    walls = {
        c1: [(r, 0) for r in range(1, oh - 1)],
        c2: [(r, ow - 1) for r in range(1, oh - 1)],
        c3: [(0, c) for c in range(1, ow - 1)],
        c4: [(oh - 1, c) for c in range(1, ow - 1)],
    }
    for col, cells in walls.items():
        for r, c in cells:
            go[r, c] = col
    objB = [(r, c, int(go[r, c])) for r in range(oh) for c in range(ow)]
    bounds = {(r, c) for r in range(oh - 2) for c in range(ow - 2)}
    objA = {random.choice(sorted(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))

    def grow():
        cand = set()
        for x in objA:
            cand |= nbrs(x)
        objA.add(random.choice(sorted((bounds - objA) & cand)))

    for _ in range(ncells - 1):
        grow()

    def shp():
        rs = [r for r, _ in objA]; cs = [c for _, c in objA]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp() != (oh - 2, ow - 2):
        grow()
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    plcdi = {(r + loci, c + locj) for r, c, _ in objB}
    rems = [(r, c) for r in range(h) for c in range(w)
            if (r, c) not in plcdi and loci + oh <= r <= h - oh + 2 and c <= w - ow + 2]
    lr, lc = random.choice(rems)
    for r, c, col in objB:
        gi[r + loci, c + locj] = col
    for r, c in objA:
        if 0 <= r + lr < h and 0 <= c + lc < w:
            gi[r + lr, c + lc] = pc
    for (r, c) in objA:
        ij = (r + 1, c + 1)
        manhs = [(min(abs(ij[0] - a) + abs(ij[1] - b) for a, b in cells), col)
                 for col, cells in walls.items()]
        mmh = min(m for m, _ in manhs)
        winners = [col for m, col in manhs if m == mmh]
        go[ij] = winners[0] if len(winners) == 1 else pc
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k)
    go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _parse(I, bgc):
    hor, ver, blob = [], [], []
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
        rows = {r for r, _ in cells}; cols = {c for _, c in cells}
        if len(rows) == 1 and max(cols) - min(cols) + 1 == len(cells) and len(cells) >= 2:
            hor.append((cells[0][0], min(cols), max(cols), col))
        elif len(cols) == 1 and max(rows) - min(rows) + 1 == len(cells) and len(cells) >= 2:
            ver.append((cells[0][1], min(rows), max(rows), col))
        else:
            blob.append((col, cells))
    if len(hor) != 2 or len(ver) != 2 or len(blob) != 1:
        return None
    hor.sort(); ver.sort()
    top, bot = hor; left, right = ver
    r0, c0 = top[0], left[0]
    oh, ow = bot[0] - r0 + 1, right[0] - c0 + 1
    if (top[1], top[2]) != (c0 + 1, c0 + ow - 2) or (left[1], left[2]) != (r0 + 1, r0 + oh - 2):
        return None
    pc, cells = blob[0]
    br = min(r for r, _ in cells); bc = min(c for _, c in cells)
    return {"r0": r0, "c0": c0, "oh": oh, "ow": ow, "pc": pc, "blob": cells,
            "br": br, "bc": bc, "ctop": top[3], "cbot": bot[3],
            "cleft": left[3], "cright": right[3]}


def derive_operations(I, O=None):
    I = np.asarray(I, dtype=int)
    info = None
    for bgc, _ in Counter(I.flatten().tolist()).most_common():
        info = _parse(I, bgc)
        if info is not None:
            break
    r0, c0, oh, ow, pc = info["r0"], info["c0"], info["oh"], info["ow"], info["pc"]
    blob = info["blob"]
    # the blob is carried so that its bbox top-left lands on the box interior corner (1,1)
    dr = r0 + 1 - info["br"]
    dc = c0 + 1 - info["bc"]
    ops, sels = [], []
    if pc != 0:
        # 1. slide the blob into the box: grab once, then continue with empty selections
        steps = [(21 if dr > 0 else 20)] * abs(dr) + [(22 if dc > 0 else 23)] * abs(dc)
        for i, op in enumerate(steps):
            ops.append(op)
            sels.append(sel_of(blob) if i == 0 else sel_of([]))
        # the vacated footprint lies outside the box and is discarded by the crop
    else:
        # colour 0 cannot be grabbed by Move; stamp the blob shape inside the box instead
        ops.append(0)
        sels.append(sel_of([(r + dr, c + dc) for r, c in blob]))
    # 2. keep only the box (full rectangle, background included)
    ops.append(33)
    sels.append([r0, c0, oh - 1, ow - 1])
    # 3. the box splits into four triangular sectors, one per wall; the blob cells
    #    inside each sector take that wall's colour (diagonal ties keep the blob colour)
    local = sorted((r + dr - r0, c + dc - c0) for r, c in blob)
    walls = [("top", info["ctop"]), ("bottom", info["cbot"]),
             ("left", info["cleft"]), ("right", info["cright"])]
    groups = {k: [] for k, _ in walls}
    for (lr, lc) in local:
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        if len(win) == 1:
            groups[win[0]].append((lr, lc))
    for name, col in walls:
        if groups[name]:
            ops.append(int(col))
            sels.append(sel_of(groups[name]))
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
