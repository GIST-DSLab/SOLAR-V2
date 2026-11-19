"""
ARC Task: 890034e9 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of
try:
    from dsl import *
except Exception:
    pass
try:
    from utils import *
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    markercol = random.choice(cols)
    remcols = [c for c in cols if c != markercol]
    numbgc = random.randint(1, 8)
    bgcols = random.sample(remcols, numbgc)
    return {"markercol": markercol, "bgcols": bgcols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, markercol=None, bgcols=None) -> dict:
    cols = interval(1, 10, 1)
    if markercol is None:
        markercol = choice(cols)
    if bgcols is None:
        remcols = remove(markercol, cols)
        numbgc = unifint(diff_lb, diff_ub, (1, 8))
        bgcols = sample(remcols, numbgc)
    bgcols = tuple(bgcols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = randint(2, h // 4)
    ow = randint(2, w // 4)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    obj = {(choice(bgcols), ij) for ij in inds}
    gi = paint(gi, obj)
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    blacks = sample(totuple(inds), numbl)
    gi = fill(gi, 0, blacks)
    patt = asindices(canvas(-1, (oh, ow)))
    tocover = set()
    for occ in occurrences(gi, recolor(0, patt)):
        tocover.add(choice(totuple(shift(patt, occ))))
    tocover = {(choice(bgcols), ij) for ij in tocover}
    gi = paint(gi, tocover)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // ((oh + 2) * (ow + 2)))))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        bd = shift(patt, loc)
        plcd = outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds = inds - plcd
            gi = fill(gi, 0, bd)
            go = fill(go, 0, bd)
            if succ == 1:
                gi = fill(gi, markercol, plcd)
            go = fill(go, markercol, plcd)
            loci, locj = loc
            ln1 = connect((loci - 1, locj), (loci - 1, locj + ow - 1))
            ln2 = connect((loci + oh, locj), (loci + oh, locj + ow - 1))
            ln3 = connect((loci, locj - 1), (loci + oh - 1, locj - 1))
            ln4 = connect((loci, locj + ow), (loci + oh - 1, locj + ow))
            if succ > 1:
                fixxer = {
                    (choice(bgcols), choice(totuple(xx))) for xx in [ln1, ln2, ln3, ln4]
                }
                gi = paint(gi, fixxer)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    grid = [[int(v) for v in row] for row in I.tolist()]
    ops, sels = [], []

    def box_cells(r0, c0, r1, c1):
        cells = set()
        for c in range(c0, c1 + 1):
            cells.add((r0, c)); cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0)); cells.add((r, c1))
        return cells

    def frame_candidates(g):
        out = []
        for col in range(1, 10):
            cells = {(r, c) for r in range(hi) for c in range(wi) if g[r][c] == col}
            if not cells:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if cells != box_cells(r0, c0, r1, c1):
                continue
            if any(g[r][c] != 0 for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)):
                continue
            out.append((col, r0, c0, r1, c1))
        return out

    def blocks_for(g, oh, ow):
        return [(r, c)
                for r in range(1, hi - oh)
                for c in range(1, wi - ow)
                if all(g[i][j] == 0
                       for i in range(r, r + oh)
                       for j in range(c, c + ow))]

    def select_frames(g, oh, ow, marker_block):
        used = set()
        chosen = []
        cands = blocks_for(g, oh, ow)
        for (r, c) in [marker_block] + [b for b in cands if b != marker_block]:
            ring = box_cells(r - 1, c - 1, r + oh, c + ow)
            if ring & used:
                continue
            used |= ring
            chosen.append((r, c))
        return chosen

    def find_marker(g):
        cands = frame_candidates(g)
        if not cands:
            return None
        scored = []
        for (col, r0, c0, r1, c1) in cands:
            n = len(select_frames(g, r1 - r0 - 1, c1 - c0 - 1, (r0 + 1, c0 + 1)))
            scored.append((0 if n >= 2 else 1, col, r0, c0, r1, c1))
        scored.sort()
        return scored[0][1:]

    found = find_marker(grid)
    if found is not None:
        markercol, mr0, mc0, mr1, mc1 = found
        oh = mr1 - mr0 - 1
        ow = mc1 - mc0 - 1
        if oh >= 2 and ow >= 2:
            marker_block = (mr0 + 1, mc0 + 1)
            targets = []
            for (r, c) in select_frames(grid, oh, ow, marker_block):
                if (r, c) == marker_block:
                    continue
                ring = box_cells(r - 1, c - 1, r + oh, c + ow)
                if all(grid[a][b] == markercol for (a, b) in ring):
                    continue
                targets.append((r, c))
            if targets:
                # copy the whole marker frame rectangle from the input; its hollow
                # interior is 0, so it is transparent and only the frame is stamped
                ops.append(28)
                sels.append([mr0, mc0, mr1 - mr0, mc1 - mc0])
                for (r, c) in targets:
                    # stamp the copied frame so it wraps the hidden empty box
                    ops.append(30)
                    sels.append([r - 1, c - 1, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 890034e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 890034e9"
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
                                f"for task 890034e9"
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
                    f"Failed to build a complete episode for task 890034e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"890034e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
