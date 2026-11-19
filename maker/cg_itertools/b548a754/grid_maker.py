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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    dotc = random.choice([c for c in range(10) if c != bgc])
    rest = [c for c in range(1, 10) if c not in (bgc, dotc)]
    boxc, inc = random.sample(rest, 2)
    return {"bgc": bgc, "boxc": boxc, "inc": inc, "dotc": dotc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, boxc=1, inc=2, dotc=3, **kw) -> dict:
    md = min(max_h, max_w)          # grids may be rotated, so bound both dims by the smaller limit
    h = _unifint(diff_lb, diff_ub, (5, md))
    w = _unifint(diff_lb, diff_ub, (4, md))
    hi = _unifint(diff_lb, diff_ub, (4, h - 1))
    wi = _unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - hi)
    locj = random.randint(0, w - wi)

    def draw(bottom):
        g = np.full((h, w), bgc, dtype=int)
        g[loci:bottom + 1, locj:locj + wi] = boxc
        g[loci + 1:bottom, locj + 1:locj + wi - 1] = inc
        return g

    go = draw(loci + hi - 1)
    cutoff = random.randint(loci + 2, loci + hi - 2)
    gi = draw(cutoff)
    gi[loci + hi - 1, random.randint(locj, locj + wi - 1)] = dotc
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    # --- read the scene from I: rectangular box (outline + uniform inside) and a lone dot ---
    boxc = inc = dotc = None
    r0 = c0 = r1 = c1 = None
    for cand in np.unique(I).tolist():
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
        cinc = int(inner[0, 0])
        rest = [x for x in np.unique(I).tolist() if x != cand and x != cinc]
        if len(rest) != 2:
            continue
        singles = [x for x in rest if int((I == x).sum()) == 1]
        if len(singles) != 1:
            continue
        boxc, inc, dotc = cand, cinc, singles[0]
        r0, c0, r1, c1 = a0, b0, a1, b1
        break
    rd, cd = [int(v) for v in np.argwhere(I == dotc)[0]]

    ops, sels = [], []
    # The wall facing the dot, the move direction, the box cross-section (a full 1-wide
    # rectangle: wall-inside-wall) and the rows/cols of the gap the wall opens.
    if cd > c1:
        wall = [(r, c1) for r in range(r0, r1 + 1)]; mop, steps = 22, cd - c1
        xsec = [r0, c1 - 1, r1 - r0, 0]
        targets = [[r0, c, 0, 0] for c in range(c1, cd)]
    elif cd < c0:
        wall = [(r, c0) for r in range(r0, r1 + 1)]; mop, steps = 23, c0 - cd
        xsec = [r0, c0 + 1, r1 - r0, 0]
        targets = [[r0, c, 0, 0] for c in range(c0, cd, -1)]
    elif rd > r1:
        wall = [(r1, c) for c in range(c0, c1 + 1)]; mop, steps = 21, rd - r1
        xsec = [r1 - 1, c0, 0, c1 - c0]
        targets = [[r, c0, 0, 0] for r in range(r1, rd)]
    else:
        wall = [(r0, c) for c in range(c0, c1 + 1)]; mop, steps = 20, r0 - rd
        xsec = [r0 + 1, c0, 0, c1 - c0]
        targets = [[r, c0, 0, 0] for r in range(r0, rd, -1)]

    if boxc != 0 and inc != 0:
        # 1. the facing wall slides out until it lands on the dot's line (grab once, then empties)
        ops.append(mop); sels.append(sel_of(wall))
        for _ in range(steps - 1):
            ops.append(mop); sels.append(sel_of([]))
        # 2. the box cross-section is copied from the input and repeated across the opened gap,
        #    starting at the wall's old line (whose 0-footprint the first copy covers)
        ops.append(28); sels.append(xsec)          # full 1-wide rectangle of the box
        for t in targets:
            ops.append(30); sels.append(t)
    else:
        # 0 is real box content: Move/Copy would drop it, so paint the stretch directly
        nr, nc = xsec[2] + 1, xsec[3] + 1
        line = [(xsec[0] + i, xsec[1] + j) for i in range(nr) for j in range(nc)]
        cur = I.copy()
        for t in targets:
            dr, dc = t[0] - xsec[0], t[1] - xsec[1]
            for col in (inc, boxc):
                cells = [(r + dr, c + dc) for r, c in line
                         if I[r, c] == col and cur[r + dr, c + dc] != col]
                if cells:
                    ops.append(int(col)); sels.append(sel_of(cells))
                    for r, c in cells:
                        cur[r, c] = col
        wr = [(r + (rd - wall[0][0] if mop in (20, 21) else 0),
               c + (cd - wall[0][1] if mop in (22, 23) else 0)) for r, c in wall]
        ops.append(int(boxc)); sels.append(sel_of(wr))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
