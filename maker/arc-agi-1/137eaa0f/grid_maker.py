"""
ARC Task: 137eaa0f (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(cols)
    dotc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "dotc": dotc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, dotc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
    remcols = remove(bgc, cols)
    remcols = remove(dotc, remcols)
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    lob = min(min(h, w) * 2, max_h)
    lobw = min(min(h, w) * 2, max_w)
    gih = unifint(diff_lb, diff_ub, (lob, max_h))
    giw = unifint(diff_lb, diff_ub, (lobw, max_w))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]})))
        for c in choscols
    )
    maxtr = min(h, w) * 2
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            break
        maxtr = int(maxtr * 1.5) + 1
        gih = randint(gih, max_h)
        giw = randint(giw, max_w)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background = the canvas colour the generator paints before scattering objects
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- count univalued 4-connected components per foreground colour ----
    seen = np.zeros((hi, wi), dtype=bool)
    comp_count = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comp_count[col] = comp_count.get(col, 0) + 1
    fg_colors = sorted(comp_count)
    cellcount = {col: int((I == col).sum()) for col in fg_colors}

    # ---- rebuild the output by stacking every colour-object on its nearest dot ----
    def reconstruct(dotc):
        dots = [(int(r), int(c)) for r, c in zip(*np.where(I == dotc))]
        if not dots:
            return None
        objs = []
        for col in fg_colors:
            if col == dotc:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]
            dot = min(dots, key=lambda d: (min(abs(d[0] - p[0]) + abs(d[1] - p[1]) for p in cells), d))
            objs.append((col, [(p[0] - dot[0], p[1] - dot[1]) for p in cells]))
        rels = [(0, 0)] + [p for _, rel in objs for p in rel]
        minr = min(p[0] for p in rels)
        minc = min(p[1] for p in rels)
        dpos = (0 - minr, 0 - minc)
        cellmap = {dpos: dotc}
        placed = []
        for col, rel in objs:
            pts = sorted((p[0] - minr, p[1] - minc) for p in rel)
            for p in pts:
                cellmap[p] = col
            placed.append((col, pts))
        H = max(p[0] for p in cellmap) + 1
        W = max(p[1] for p in cellmap) + 1
        if (H, W) != (ho, wo):
            return None
        g = np.full((H, W), -1, dtype=int)
        for (r, c), v in cellmap.items():
            g[r, c] = v
        if not np.array_equal(g, O):
            return None
        return dotc, dpos, placed

    # dot colour = colour present in every object (most components; ties -> fewest cells)
    cands = sorted(fg_colors, key=lambda col: (-comp_count[col], cellcount[col], col))
    res = None
    for cand in cands:
        res = reconstruct(cand)
        if res is not None:
            break

    ops, sels = [], []
    if res is None:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    dotc, dpos, placed = res

    # build the stacked picture in the top-left region, source (input) stays readable
    # 1) the shared reference dot
    if int(I[dpos[0], dpos[1]]) != dotc:
        ops.append(int(dotc))
        sels.append([dpos[0], dpos[1], 0, 0])

    # 2) one colour-object at a time, laid down around that dot
    placed.sort(key=lambda t: (-len(t[1]), t[0]))
    for col, pts in placed:
        pts = sorted(pts)
        i = 0
        while i < len(pts):
            r, c = pts[i]
            if int(I[r, c]) == col:
                i += 1
                continue
            j = i
            end = c
            while (j + 1 < len(pts) and pts[j + 1][0] == r and pts[j + 1][1] == end + 1
                   and int(I[pts[j + 1][0], pts[j + 1][1]]) != col):
                j += 1
                end = pts[j][1]
            ops.append(int(col))
            sels.append([r, c, 0, end - c])
            i = j + 1

    # 3) only now shrink the canvas onto the finished picture
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 137eaa0f"
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
                                f"for task 137eaa0f"
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
                    f"Failed to build a complete episode for task 137eaa0f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"137eaa0f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
