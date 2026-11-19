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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, occcol = sample(cols, 3)
    return {"bgc": bgc, "objc": objc, "occcol": occcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, occcol) -> dict:
    h_lo = min(10, max_h)
    w_lo = min(10, max_w)
    h = unifint(diff_lb, diff_ub, (h_lo, max_h))
    w = unifint(diff_lb, diff_ub, (w_lo, max_w))
    oh = unifint(diff_lb, diff_ub, (4, max(4, h - 2)))
    ow = unifint(diff_lb, diff_ub, (4, max(4, (w - 2) // 2)))
    oh = min(oh, h - 2)
    ow = min(ow, (w - 2) // 2)
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        opts = totuple((bounds - shp) & mapply(neighbors, shp))
        if len(opts) == 0:
            break
        shp.add(choice(opts))
    while height(shp) < 3 or width(shp) < 3:
        opts = totuple((bounds - shp) & mapply(neighbors, shp))
        if len(opts) == 0:
            break
        shp.add(choice(opts))
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
    boxw = unifint(diff_lb, diff_ub, (2, max(2, ow // 2)))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow // 2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    transposers = (dmirror, cmirror, rot90, rot270)
    nmfs = choice((1, 2))
    fns = list(sample(mfs, nmfs))
    ntrans = sum(1 for f in fns if f in transposers) % 2
    if ntrans == 1 and (w > max_h or h > max_w):
        fns = [f for f in fns if f not in transposers]
        if len(fns) == 0:
            fns = [identity]
    for fn in fns:
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    A = np.asarray(I, dtype=int)
    h, w = A.shape
    G = [[int(v) for v in row] for row in A.tolist()]

    # background = dominant colour on the frame (objects never touch all borders)
    border = [G[0][c] for c in range(w)] + [G[h - 1][c] for c in range(w)] \
           + [G[r][0] for r in range(h)] + [G[r][w - 1] for r in range(h)]
    bgc = Counter(border).most_common(1)[0][0]

    cellmap = {}
    for r in range(h):
        row = G[r]
        for c in range(w):
            cellmap.setdefault(row[c], set()).add((r, c))

    fg = sorted(k for k in cellmap if k != bgc)
    ops, sels = [], []
    if len(fg) != 2:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    def is_solid(cs):
        rs = [r for r, _ in cs]; cls = [c for _, c in cs]
        return len(cs) == (max(rs) - min(rs) + 1) * (max(cls) - min(cls) + 1)

    # the occluding patch is the colour whose cells form a full solid rectangle
    cands = [c for c in fg if is_solid(cellmap[c])]
    if not cands:
        cands = list(fg)

    def search(objc, occc):
        """Find the mirror (left-right or up-down) + offset that maps the visible
        figure onto itself without ever landing on background: that reflection is
        the figure's symmetry, and it reconstructs whatever the patch hides."""
        obj = cellmap[objc]; occ = cellmap[occc]
        rs = [r for r, _ in obj]; cs = [c for _, c in obj]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        mirrors = [[(r, c0 + c1 - c) for r, c in obj],
                   [(r0 + r1 - r, c) for r, c in obj]]
        comb = obj | occ
        rs2 = [r for r, _ in comb]; cs2 = [c for _, c in comb]
        H = max(rs2) - min(rs2) + 1
        W = max(cs2) - min(cs2) + 1
        k = max(H // 2 + 1, W // 2 + 1)
        best = None
        for m in mirrors:
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    ok = True
                    sc = 0
                    for (r, c) in m:
                        rr = r + dr; cc = c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            v = G[rr][cc]
                            if v == bgc:
                                ok = False
                                break
                            if v == objc:
                                sc += 1
                    if not ok:
                        continue
                    key = (sc, -(abs(dr) + abs(dc)))
                    if best is None or key > best[0]:
                        best = (key, [(r + dr, c + dc) for r, c in m])
        return best

    chosen = None
    for occc in cands:
        objc = fg[0] if fg[1] == occc else fg[1]
        b = search(objc, occc)
        if b is None:
            continue
        ratio = b[0][0] / float(max(1, len(cellmap[objc])))
        key = (ratio, -len(cellmap[occc]))
        if chosen is None or key > chosen[0]:
            chosen = (key, objc, occc, b[1])

    if chosen is None:
        occc = cands[0]
        objc = fg[0] if fg[1] == occc else fg[1]
        reflected = []
    else:
        _, objc, occc, reflected = chosen

    patch = sorted(cellmap[occc])
    # 1. erase the occluding patch back to background (base layer)
    ops.append(int(bgc)); sels.append(sel_of(patch))

    # 2. draw back the piece of the symmetric figure that the patch was hiding
    restore = sorted(set(reflected) & cellmap[occc])
    if restore:
        ops.append(int(objc)); sels.append(sel_of(restore))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # full grid rectangle
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
