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
import numpy as np
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, a + round((b - a) * d)))


def sample_colors(num_examples=None) -> dict:
    bgc, pc, c1, c2, c3, c4 = random.sample(range(10), 6)
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


def _rot90(g):
    return [list(r) for r in zip(*g[::-1])]


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4, **kw) -> dict:
    h = _unifint(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unifint(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unifint(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unifint(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * ow for _ in range(oh)]
    walls = {
        c1: [(i, 0) for i in range(1, oh - 1)],
        c2: [(i, ow - 1) for i in range(1, oh - 1)],
        c3: [(0, j) for j in range(1, ow - 1)],
        c4: [(oh - 1, j) for j in range(1, ow - 1)],
    }
    for col, cells in walls.items():
        for i, j in cells:
            go[i][j] = col
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

    def shp(s):
        rs = [i for i, _ in s]; cs = [j for _, j in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(objA) != (oh - 2, ow - 2):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    plcdi = {(i + loci, j + locj) for i in range(oh) for j in range(ow) if go[i][j] != bgc}
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in plcdi and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    for i in range(oh):
        for j in range(ow):
            if go[i][j] != bgc:
                gi[i + loci][j + locj] = go[i][j]
    for (i, j) in objA:
        gi[i + li][j + lj] = pc
    for (i, j) in objA:
        ii, jj = i + 1, j + 1
        d = {col: min(abs(ii - a) + abs(jj - b) for a, b in cells) for col, cells in walls.items()}
        mn = min(d.values())
        win = [c for c, v in d.items() if v == mn]
        go[ii][jj] = win[0] if len(win) == 1 else pc
    k = random.randint(0, 3)
    for _ in range(k):
        gi = _rot90(gi); go = _rot90(go)
    return {"input": gi, "output": go}


def _parse(I):
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    by_col = {}
    for (r, c), v in np.ndenumerate(I):
        if v != bgc:
            by_col.setdefault(int(v), []).append((r, c))
    pc = None
    for col, cells in by_col.items():
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        if max(rs) > min(rs) and max(cs) > min(cs):   # the blob spans >=2 rows and cols; walls are lines
            pc = col
    blob = by_col[pc]
    wall_cells = [rc for col, cells in by_col.items() if col != pc for rc in cells]
    r0 = min(r for r, _ in wall_cells); c0 = min(c for _, c in wall_cells)
    oh = max(r for r, _ in wall_cells) - r0 + 1
    ow = max(c for _, c in wall_cells) - c0 + 1
    br = min(r for r, _ in blob); bc = min(c for _, c in blob)
    return dict(bgc=bgc, pc=pc, blob=blob, r0=r0, c0=c0, oh=oh, ow=ow, br=br, bc=bc,
                top=int(I[r0, c0 + 1]), bottom=int(I[r0 + oh - 1, c0 + 1]),
                left=int(I[r0 + 1, c0]), right=int(I[r0 + 1, c0 + ow - 1]))


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    info = _parse(I)
    r0, c0, oh, ow = info["r0"], info["c0"], info["oh"], info["ow"]
    br, bc, pc = info["br"], info["bc"], info["pc"]
    ops, sels = [], []

    # 1. the box is the answer's frame: crop the canvas to it (full rectangle, bg included)
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])

    # 2. carry the blob into the box as ONE unit: its bbox is exactly the box interior,
    #    so copy that full rectangle from the input and paste it at the interior corner (1,1)
    landed = [(r - br + 1, c - bc + 1) for (r, c) in info["blob"]]
    if pc != 0:
        ops.append(28); sels.append([br, bc, oh - 3, ow - 3])   # whole blob bbox rectangle
        ops.append(30); sels.append([1, 1, 0, 0])
    else:
        # colour 0 cannot travel through the clipboard; stamp the landed cells directly
        ops.append(0); sels.append(sel_of(landed))

    # 3. 45-degree cuts from the four corners split the interior into four wedges, one per
    #    wall; blob cells inside a wedge take that wall's colour, cells on a cut stay as they are
    def wedge(lr, lc):
        d = {"top": lr - 1, "bottom": oh - 2 - lr, "left": lc - 1, "right": ow - 2 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        return win[0] if len(win) == 1 else None

    for name in ("top", "bottom", "left", "right"):
        grp = [(lr, lc) for (lr, lc) in landed if wedge(lr, lc) == name]
        if grp:
            ops.append(info[name]); sels.append(sel_of(grp))

    ops.append(34); sels.append([0, 0, oh - 1, ow - 1])
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
