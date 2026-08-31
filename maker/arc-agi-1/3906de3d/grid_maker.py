"""
ARC Task: 3906de3d (RE-ARC) — LLM-generated grid_maker
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

ROTS = ['identity', 'rot90', 'rot180', 'rot270']


def sample_colors(num_examples=None) -> dict:
    # linc must be non-zero: the line is the object that gets MOVED, and ARCLE's
    # object buffer keeps only non-zero cells (a 0-coloured line would be a NOOP).
    linc = random.choice([c for c in range(10) if c != 0])
    rest = [c for c in range(10) if c != linc]
    bgc, boxc = random.sample(rest, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rotname": r} for r in ROTS]
        examples += [{"rotname": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rotname": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "boxc": boxc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, boxc, linc, rotname=None) -> dict:
    if rotname is None:
        rotname = choice(tuple(ROTS))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    oh = unifint(diff_lb, diff_ub, (2, h // 2))
    ow = unifint(diff_lb, diff_ub, (3, w - 2))
    locj = randint(1, w - ow - 1)
    bx = backdrop(frozenset({(0, locj), (oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, boxc, bx)
    rng = range(locj, locj + ow)
    cutoffs = [randint(1, oh - 1) for j in rng]
    for jj, co in zip(rng, cutoffs):
        gi = fill(gi, bgc, connect((co, jj), (oh - 1, jj)))
    numlns = unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = sample(list(rng), numlns)
    go = tuple(e for e in gi)
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = randint(1, h - co - 1)
            linei = connect((h - lineh, jj), (h - 1, jj))
            lineo = connect((co, jj), (co + lineh - 1, jj))
            gi = fill(gi, linc, linei)
            go = fill(go, linc, lineo)
    rotf = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}[rotname]
    return {'input': rotf(gi), 'output': rotf(go)}


def derive_operations(I, O):
    # Rule: a comb-shaped BOX hangs off one edge (its teeth have varying depth).
    # Free LINES sit on the OPPOSITE edge, each inside one of the box's lanes.
    # Each line slides across the gap until it butts against its lane's tooth.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # The box never reaches a corner (locj>=1, oh<=h//2) and neither do the lines,
    # so a corner cell is always the background colour, in any rotation.
    bgc = int(I[0, 0])
    others = sorted({int(v) for v in np.unique(I)} - {bgc})
    if len(others) < 2:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    c1, c2 = others[0], others[1]

    def touches(c):
        m = (I == c)
        e = set()
        if m[0, :].any(): e.add('T')
        if m[hi - 1, :].any(): e.add('B')
        if m[:, 0].any(): e.add('L')
        if m[:, wi - 1].any(): e.add('R')
        return e

    # Box and lines cling to opposite edges -> that pair fixes the sliding axis.
    vertical = bool({'T', 'B'} & touches(c1))

    def lanes_used(c):
        m = (I == c)
        return int(m.any(axis=0).sum()) if vertical else int(m.any(axis=1).sum())

    # The box spans every lane (ow of them); the lines occupy a strict subset.
    boxc, linc = (c1, c2) if lanes_used(c1) > lanes_used(c2) else (c2, c1)
    ebox = touches(boxc)
    low = ('T' in ebox) if vertical else ('L' in ebox)

    n = hi if vertical else wi          # length along the sliding axis
    nlanes = wi if vertical else hi
    move_op = (20 if low else 21) if vertical else (23 if low else 22)

    for p in range(nlanes):
        lane = [int(I[u, p]) if vertical else int(I[p, u]) for u in range(n)]
        if linc not in lane:
            continue
        if low:                                   # box at u=0, line at u=n-1
            lineh = 0; u = n - 1
            while u >= 0 and lane[u] == linc:
                lineh += 1; u -= 1
            tooth = 0; u = 0
            while u < n and lane[u] == boxc:
                tooth += 1; u += 1
            src, dst = n - lineh, tooth
        else:                                     # box at u=n-1, line at u=0
            lineh = 0; u = 0
            while u < n and lane[u] == linc:
                lineh += 1; u += 1
            tooth = 0; u = n - 1
            while u >= 0 and lane[u] == boxc:
                tooth += 1; u -= 1
            src, dst = 0, n - tooth - lineh
        if lineh == 0 or src == dst:
            continue

        # slide this whole line, one cell per step, until it meets the tooth
        shift = abs(dst - src)
        cur = src
        for _ in range(shift):
            sels.append([cur, p, lineh - 1, 0] if vertical else [p, cur, 0, lineh - 1])
            ops.append(move_op)
            cur += (-1 if low else 1)

        # the slide leaves the swept-through strip at 0; restore it to background
        if bgc != 0:
            if low:
                a, b = dst + lineh, n - 1
            else:
                a, b = 0, dst - 1
            if a <= b:
                sels.append([a, p, b - a, 0] if vertical else [p, a, 0, b - a])
                ops.append(bgc)

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3906de3d"
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
                                f"for task 3906de3d"
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
                    f"Failed to build a complete episode for task 3906de3d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3906de3d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
