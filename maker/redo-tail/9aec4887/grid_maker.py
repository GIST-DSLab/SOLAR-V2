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

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # pragma: no cover
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


# --------------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    """bgc, pc and the four wall colours are one fixed 6-colour scheme per episode.
    The generator also picks a random rotation (identity/90/180/270) — a discrete
    structural variant, so it is planned per instance instead of resampled."""
    cols = list(range(10))
    bgc, pc, c1, c2, c3, c4 = random.sample(cols, 6)
    ROTS = [0, 1, 2, 3]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]            # test rotation was shown
    return {"bgc": bgc, "pc": pc, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "instance_plan": plan}


# ------------------------------------------------------------------- generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, pc, c1, c2, c3, c4,
             rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        return random.randint(lb + int((ub - lb) * diff_lb),
                              lb + int((ub - lb) * diff_ub))

    if rot in (1, 3):                                # rotation swaps h and w
        hcap = wcap = min(max_h, max_w)
    else:
        hcap, wcap = max_h, max_w
    hcap = max(12, min(30, hcap))
    wcap = max(12, min(30, wcap))

    h = unifint(12, hcap)
    w = unifint(12, wcap)
    oh = unifint(4, h // 2 - 2)
    ow = unifint(4, w // 2 - 2)

    # the box: four coloured walls, background corners
    go = [[bgc] * ow for _ in range(oh)]
    for i in range(1, oh - 1):
        go[i][0] = c1
        go[i][ow - 1] = c2
    for j in range(1, ow - 1):
        go[0][j] = c3
        go[oh - 1][j] = c4

    # the blob: 8-connected, spanning exactly the (oh-2, ow-2) interior
    ih, iw = oh - 2, ow - 2
    bounds = [(i, j) for i in range(ih) for j in range(iw)]
    objA = {random.choice(bounds)}
    ncells = unifint(1, max(1, (ih * iw) // 2))

    def frontier(s):
        cand = set()
        for (i, j) in s:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    p = (i + di, j + dj)
                    if 0 <= p[0] < ih and 0 <= p[1] < iw and p not in s:
                        cand.add(p)
        return sorted(cand)

    def shape_of(s):
        rs = [p[0] for p in s]
        cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    for _ in range(ncells - 1):
        f = frontier(objA)
        if not f:
            break
        objA.add(random.choice(f))
    while shape_of(objA) != (ih, iw):
        f = frontier(objA)
        if not f:
            break
        objA.add(random.choice(f))

    # placement: box somewhere, blob strictly below it (never overlapping)
    gi = [[bgc] * w for _ in range(h)]
    loci = random.randint(0, h - 2 * oh + 2)
    locj = random.randint(0, w - ow)
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = go[i][j]
    rems = [(i, j) for i in range(loci + oh, h - oh + 3)
            for j in range(0, w - ow + 3)]
    lr, lc = random.choice(rems)
    for (i, j) in objA:
        gi[lr + i][lc + j] = pc

    # each blob cell takes the colour of its nearest wall; ties keep the blob colour
    for (i, j) in objA:
        r, c = i + 1, j + 1
        d = [(r, c3), (oh - 1 - r, c4), (c, c1), (ow - 1 - c, c2)]
        mn = min(x[0] for x in d)
        hits = [col for dd, col in d if dd == mn]
        go[r][c] = hits[0] if len(hits) == 1 else pc

    def rotate(g, k):
        g = [list(row) for row in g]
        for _ in range(k):
            g = [list(row) for row in zip(*g[::-1])]
        return g

    gi = rotate(gi, rot)
    go = rotate(go, rot)
    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# --------------------------------------------------------------------- derive
def _parse(I, bgc):
    """Split the non-background content into the 4 straight walls of the box
    (each 1 cell thick) and the single 2-D blob."""
    hi, wi = I.shape
    cells_by_color = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells_by_color.setdefault(v, []).append((r, c))
    if len(cells_by_color) != 5:
        return None
    hor, ver, blob = [], [], None
    for col, cells in cells_by_color.items():
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        bh = max(rs) - min(rs) + 1
        bw = max(cs) - min(cs) + 1
        if bh == 1 and bw >= 2 and len(cells) == bw:
            hor.append((min(rs), min(cs), max(cs), col))
        elif bw == 1 and bh >= 2 and len(cells) == bh:
            ver.append((min(cs), min(rs), max(rs), col))
        elif bh >= 2 and bw >= 2:
            if blob is not None:
                return None
            blob = (col, cells, min(rs), min(cs), bh, bw)
        else:
            return None
    if len(hor) != 2 or len(ver) != 2 or blob is None:
        return None
    hor.sort()
    ver.sort()
    top, bot = hor
    left, right = ver
    r0, r1 = top[0], bot[0]
    c0, c1 = left[0], right[0]
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    if oh < 4 or ow < 4:
        return None
    if (top[1], top[2]) != (c0 + 1, c1 - 1) or (bot[1], bot[2]) != (c0 + 1, c1 - 1):
        return None
    if (left[1], left[2]) != (r0 + 1, r1 - 1) or (right[1], right[2]) != (r0 + 1, r1 - 1):
        return None
    if (blob[4], blob[5]) != (oh - 2, ow - 2):       # blob exactly fills the interior
        return None
    return dict(r0=r0, c0=c0, oh=oh, ow=ow,
                ctop=top[3], cbot=bot[3], cleft=left[3], cright=right[3],
                pc=blob[0], blob=blob[1], br=blob[2], bc=blob[3])


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)

    info = None
    for bgc, _ in Counter(I.flatten().tolist()).most_common():
        info = _parse(I, bgc)
        if info is not None:
            break

    r0, c0 = info["r0"], info["c0"]
    oh, ow = info["oh"], info["ow"]
    pc = info["pc"]

    # the blob, carried into the box interior (its top-left lands on (1,1) of the box)
    interior = []
    for (r, c) in info["blob"]:
        lr = r - info["br"] + 1
        lc = c - info["bc"] + 1
        interior.append((r0 + lr, c0 + lc, lr, lc))
    interior.sort()

    walls = [("top", info["ctop"]), ("bottom", info["cbot"]),
             ("left", info["cleft"]), ("right", info["cright"])]
    claim = {}
    for (gr, gc, lr, lc) in interior:
        d = {"top": lr, "bottom": oh - 1 - lr, "left": lc, "right": ow - 1 - lc}
        mn = min(d.values())
        winners = [k for k, v in d.items() if v == mn]
        claim[(gr, gc)] = winners[0] if len(winners) == 1 else None

    ops, sels = [], []

    # 1. stamp the whole blob inside the box, in its own colour (the base layer)
    ops.append(int(pc))
    sels.append(sel_of([(gr, gc) for (gr, gc, _, _) in interior]))

    # 2. each wall repaints the blob cells that are strictly closest to it;
    #    the cells left in the blob colour are the ones tied between two walls
    for name, col in walls:
        grp = [(gr, gc) for (gr, gc, _, _) in interior if claim[(gr, gc)] == name]
        if grp:
            ops.append(int(col))
            sels.append(sel_of(grp))

    # 3. keep only the box — selection is exactly that full rectangle, background included
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
