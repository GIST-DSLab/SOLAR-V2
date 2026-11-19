"""
ARC Task: 3345333e (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    bgc, objc, occcol = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "occcol": occcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, occcol) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow // 2))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading taken from the golf entry: the rectangle is not a hole to patch with a copied
    region -- it is noise laid over a mirror-symmetric blob.  Remove it (paint it background);
    the answer is then the blob united with its own mirror image: every cell whose mirror
    partner is blob-coloured becomes blob-coloured.
      1) erase the occluder (all cells of its colour),
      2) paint the reflected blob cells that do not already hold the blob colour.
    The axis is measured from I alone: the reflection of the visible blob that lands only on
    blob/occluder cells and overlaps the visible blob the most."""
    import numpy as np
    from maker.sel_helpers import sel_of

    G = np.asarray(I, dtype=int)
    H, W = G.shape
    g = [[int(x) for x in row] for row in G]
    border = ([g[0][j] for j in range(W)] + [g[H - 1][j] for j in range(W)]
              + [g[i][0] for i in range(H)] + [g[i][W - 1] for i in range(H)])
    bg = max(sorted(set(border)), key=border.count)          # canvas colour: the frame
    cellsof = {}
    for i in range(H):
        for j in range(W):
            cellsof.setdefault(g[i][j], []).append((i, j))
    # occluder: the non-background colour whose cells fill their bounding box
    occc = None; occ_area = None
    for col in sorted(cellsof):
        if col == bg:
            continue
        st = cellsof[col]
        rs = [p[0] for p in st]; cs = [p[1] for p in st]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        if len(st) == area and (occ_area is None or area < occ_area):
            occc = col; occ_area = area
    oc = [col for col in sorted(cellsof) if col not in (bg, occc)][0]   # blob colour
    V = cellsof[oc]; B = cellsof[occc]
    Vs = set(V); Bs = set(B); BGs = set(cellsof[bg])

    comb = V + B
    hh = max(p[0] for p in comb) - min(p[0] for p in comb) + 1
    ww = max(p[1] for p in comb) - min(p[1] for p in comb) + 1
    k = max(hh // 2 + 1, ww // 2 + 1)
    vD = min(p[1] for p in V) + max(p[1] for p in V)
    hD = min(p[0] for p in V) + max(p[0] for p in V)

    def reflect(mt, di, dj, r, c):
        return (r + di, vD - c + dj) if mt == 'v' else (hD - r + di, c + dj)

    best = None; bestsc = -1
    for mt in ('v', 'h'):
        for di in range(-k, k + 1):
            for dj in range(-k, k + 1):
                M = [reflect(mt, di, dj, r, c) for r, c in V]
                if any(p in BGs for p in M):
                    continue
                sc = sum(1 for p in M if p in Vs)
                if sc > bestsc:
                    bestsc = sc; best = (mt, di, dj)
    mt, di, dj = best

    ops, sels = [], []
    # 1. the occluding rectangle is not part of the picture: erase it (all cells of its colour)
    ops.append(bg); sels.append(sel_of(sorted(Bs)))
    # 2. the blob's mirror image: paint the reflected blob cells not already blob-coloured
    mirror = {reflect(mt, di, dj, r, c) for r, c in V}
    todo = sorted(p for p in mirror
                  if 0 <= p[0] < H and 0 <= p[1] < W and p not in Vs)
    if todo:
        ops.append(oc); sels.append(sel_of(todo))
    ops.append(34); sels.append(sel_of([(r, c) for r in range(H) for c in range(W)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 3345333e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3345333e"
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
                                f"for task 3345333e"
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
                    f"Failed to build a complete episode for task 3345333e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3345333e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
