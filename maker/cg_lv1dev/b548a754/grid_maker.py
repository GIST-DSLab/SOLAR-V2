"""
ARC Task: b548a754 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical scene


def sample_colors(num_examples=None) -> dict:
    bgc, boxc, inc, dotc = random.sample(range(10), 4)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, inc, dotc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return int(round(random.uniform(lo, hi_)))

    if rot is None:
        rot = random.choice(ROTS)
    lim = min(max_h, max_w) if rot in (1, 3) else None
    mh, mw = (lim, lim) if lim else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (5, max(5, mh)))
    w = unifint(diff_lb, diff_ub, (4, max(4, mw)))
    hi = unifint(diff_lb, diff_ub, (4, h - 1))
    wi = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)

    def draw(g, r0, c0, r1, c1):
        g[r0:r1 + 1, c0:c1 + 1] = boxc
        g[r0 + 1:r1, c0 + 1:c1] = inc

    go = np.full((h, w), bgc, dtype=int)
    draw(go, loci, locj, loci + hi - 1, locj + wi - 1)
    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = np.full((h, w), bgc, dtype=int)
    draw(gi, loci, locj, cutoff, locj + wi - 1)
    gi[loci + hi - 1, random.randint(locj, locj + wi - 1)] = dotc
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # --- read the scene from I: outlined box with uniform interior, plus a lone dot ---
    boxc = inc = dotc = None
    r0 = c0 = r1 = c1 = None
    vals = np.unique(I).tolist()
    for cand in vals:
        cells = np.argwhere(I == cand)
        a0, b0 = int(cells[:, 0].min()), int(cells[:, 1].min())
        a1, b1 = int(cells[:, 0].max()), int(cells[:, 1].max())
        bh, bw = a1 - a0 + 1, b1 - b0 + 1
        if bh < 3 or bw < 3 or len(cells) != 2 * bh + 2 * bw - 4:
            continue
        sub = I[a0:a1 + 1, b0:b1 + 1]
        if not (np.all(sub[0, :] == cand) and np.all(sub[-1, :] == cand)
                and np.all(sub[:, 0] == cand) and np.all(sub[:, -1] == cand)):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or len(np.unique(inner)) != 1 or int(inner[0, 0]) == cand:
            continue
        ci = int(inner[0, 0])
        rest = [x for x in vals if x != cand and x != ci]
        if len(rest) != 2:
            continue
        singles = [x for x in rest if int((I == x).sum()) == 1]
        if len(singles) != 1:
            continue
        boxc, inc, dotc = cand, ci, singles[0]
        r0, c0, r1, c1 = a0, b0, a1, b1
        break

    rd, cd = [int(v) for v in np.argwhere(I == dotc)[0]]

    # --- the wall facing the dot, its direction of travel, and the interior cross-section ---
    if cd > c1:
        d, mop = (0, 1), 22
        wall = [(r, c1) for r in range(r0, r1 + 1)]; dist = cd - c1
        cross = [(r, c1 - 1) for r in range(r0, r1 + 1)]
    elif cd < c0:
        d, mop = (0, -1), 23
        wall = [(r, c0) for r in range(r0, r1 + 1)]; dist = c0 - cd
        cross = [(r, c0 + 1) for r in range(r0, r1 + 1)]
    elif rd > r1:
        d, mop = (1, 0), 21
        wall = [(r1, c) for c in range(c0, c1 + 1)]; dist = rd - r1
        cross = [(r1 - 1, c) for c in range(c0, c1 + 1)]
    else:
        d, mop = (-1, 0), 20
        wall = [(r0, c) for c in range(c0, c1 + 1)]; dist = r0 - rd
        cross = [(r0 + 1, c) for c in range(c0, c1 + 1)]

    def shift(cells, k):
        return [(r + k * d[0], c + k * d[1]) for r, c in cells]

    G = I.copy()
    ops, sels = [], []

    def paint(col, cells):
        cells = list(cells)
        if cells and any(G[r, c] != col for r, c in cells):
            ops.append(int(col)); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = col

    # 1. the facing wall slides out, one cell at a time, until it lands on the dot's line
    dest = shift(wall, dist)
    if boxc != 0:
        ops.append(mop); sels.append(sel_of(wall))          # grab the wall
        for _ in range(dist - 1):
            ops.append(mop); sels.append(sel_of([]))       # keep sliding same object
        for r, c in wall:
            G[r, c] = 0                                     # original footprint left at 0
        for r, c in dest:
            G[r, c] = boxc
    else:
        paint(boxc, dest)                                   # a 0-coloured wall can't be grabbed

    # 2. the body stretches: replicate the interior cross-section across the gap it left
    gaps = [shift(wall, k) for k in range(dist)]            # old wall line .. line before new wall
    if boxc != 0 and inc != 0:
        rr = [r for r, _ in cross]; cc = [c for _, c in cross]
        ops.append(28); sels.append(sel_of(cross))          # CopyI one interior slice
        for line in gaps:
            tr = min(r for r, _ in line); tc = min(c for _, c in line)
            ops.append(30); sels.append([tr, tc, 0, 0])     # paste slice onto each gap line
            for (sr, sc), (gr, gc) in zip(cross, line):
                G[gr, gc] = I[sr, sc]
    else:
        band = [cell for line in gaps for cell in line[1:-1]]
        sides = [cell for line in gaps for cell in (line[0], line[-1])]
        paint(inc, band)
        paint(boxc, sides)

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b548a754"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b548a754"
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
                                f"for task b548a754"
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
                    f"Failed to build a complete episode for task b548a754 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b548a754-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
