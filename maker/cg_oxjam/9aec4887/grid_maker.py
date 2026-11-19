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


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4) -> dict:
    h = _unif(diff_lb, diff_ub, (12, max(12, max_h)))
    w = _unif(diff_lb, diff_ub, (12, max(12, max_w)))
    oh = _unif(diff_lb, diff_ub, (4, h // 2 - 2))
    ow = _unif(diff_lb, diff_ub, (4, w // 2 - 2))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * ow for _ in range(oh)]
    lines = {}
    lines[c1] = [(i, 0) for i in range(1, oh - 1)]
    lines[c2] = [(i, ow - 1) for i in range(1, oh - 1)]
    lines[c3] = [(0, j) for j in range(1, ow - 1)]
    lines[c4] = [(oh - 1, j) for j in range(1, ow - 1)]
    for col, cells in lines.items():
        for (i, j) in cells:
            go[i][j] = col
    bounds = {(i, j) for i in range(oh - 2) for j in range(ow - 2)}

    def nbrs(cs):
        out = set()
        for (i, j) in cs:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out

    objA = {random.choice(sorted(bounds))}
    ncells = _unif(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for _ in range(ncells - 1):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))

    def shp(o):
        rs = [a for a, _ in o]; cs = [b for _, b in o]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
    while shp(objA) != (oh - 2, ow - 2):
        objA.add(random.choice(sorted((bounds - objA) & nbrs(objA))))
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    plcd = set()
    for i in range(oh):
        for j in range(ow):
            plcd.add((i + loci, j + locj))
            gi[i + loci][j + locj] = go[i][j]
    rems = [(i, j) for i in range(h) for j in range(w)
            if (i, j) not in plcd and loci + oh <= i <= h - oh + 2 and j <= w - ow + 2]
    li, lj = random.choice(rems)
    for (i, j) in objA:
        gi[i + li][j + lj] = pc
    for (i, j) in objA:
        a, b = i + 1, j + 1
        d = {}
        for col, cells in lines.items():
            d[col] = min(abs(a - x) + abs(b - y) for x, y in cells)
        mn = min(d.values())
        win = [c for c, v in d.items() if v == mn]
        go[a][b] = win[0] if len(win) == 1 else pc
    k = random.randint(0, 3)
    gi = np.rot90(np.array(gi), -k).tolist()
    go = np.rot90(np.array(go), -k).tolist()
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    cells = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                cells.setdefault(int(I[r, c]), []).append((r, c))
    # the blob is the only non-background colour that is not a straight segment
    pc = None
    walls = []
    for col, cs in cells.items():
        rows = {r for r, _ in cs}; cols = {c for _, c in cs}
        if len(rows) > 1 and len(cols) > 1:
            pc = col
        else:
            walls.extend(cs)
    blob = cells[pc]
    r0 = min(r for r, _ in walls); c0 = min(c for _, c in walls)
    oh = max(r for r, _ in walls) - r0 + 1
    ow = max(c for _, c in walls) - c0 + 1
    wcol = {"top": int(I[r0, c0 + 1]), "bottom": int(I[r0 + oh - 1, c0 + 1]),
            "left": int(I[r0 + 1, c0]), "right": int(I[r0 + 1, c0 + ow - 1])}
    br = min(r for r, _ in blob); bc = min(c for _, c in blob)
    dr, dc = r0 + 1 - br, c0 + 1 - bc

    ops, sels = [], []
    dest = [(r + dr, c + dc) for r, c in blob]
    if pc != 0:
        # 1. carry the blob, as one object, into the box interior (grab once, then empties)
        steps = [20 if dr < 0 else 21] * abs(dr) + [23 if dc < 0 else 22] * abs(dc)
        first = True
        for op in steps:
            ops.append(op)
            sels.append(sel_of(blob) if first else sel_of([]))
            first = False
        # the vacated footprint lies outside the box and is cropped away below
    else:
        # a 0-coloured blob cannot be grabbed by Move (0 = empty); stamp it instead
        ops.append(0)
        sels.append(sel_of(dest))

    # 2. each wall claims the blob cells strictly closest to it; ties keep blob colour
    claim = {}
    for (gr, gc) in dest:
        lr, lc = gr - r0, gc - c0
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        win = [k for k, v in d.items() if v == mn]
        claim[(gr, gc)] = win[0] if len(win) == 1 else None
    for name in ("top", "bottom", "left", "right"):
        grp = sorted(p for p in dest if claim[p] == name)
        if grp:
            ops.append(wcol[name]); sels.append(sel_of(grp))

    # 3. keep only the box (selection is exactly that full rectangle, background included)
    ops.append(33); sels.append([r0, c0, oh - 1, ow - 1])
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
