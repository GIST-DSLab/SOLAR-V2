"""
ARC Task: 1caeab9d (RE-ARC) — LLM-generated grid_maker
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
    # colour 1 is the hard-coded anchor; only the background is sampled (from non-1 colours).
    # Other object colours are free: the rule depends only on colour 1 and the background.
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    H = max(3, min(30, max_h))
    W = max(6, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (3, H))
    w = unifint(diff_lb, diff_ub, (6, W))
    oh = unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
    ow = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))
    bb = {(i, j) for i in range(oh) for j in range(ow)}
    sp = random.choice(sorted(bb))
    obj = {sp}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for _ in range(ncells):
        cands = set()
        for (a, b) in obj:
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (a + da, b + db)
                if p in bb and p not in obj:
                    cands.add(p)
        if not cands:
            break
        obj.add(random.choice(sorted(cands)))
    mi = min(a for a, _ in obj); mj = min(b for _, b in obj)
    obj = {(a - mi, b - mj) for a, b in obj}
    oh = max(a for a, _ in obj) + 1
    ow = max(b for _, b in obj) + 1
    loci = random.randint(0, h - oh)
    numo = unifint(diff_lb, diff_ub, (2, max(2, min(8, w // ow)))) - 1
    locj = random.randint(0, w - ow)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    def fill(g, col, di, dj):
        for a, b in obj:
            g[a + di][b + dj] = col

    fill(gi, 1, loci, locj)
    fill(go, 1, loci, locj)
    remcols = [c for c in cols if c != bgc]
    itv = set(range(w)) - set(range(locj, locj + ow))
    for _ in range(numo):
        cands = [j for j in sorted(itv) if set(range(j, j + ow)).issubset(itv)]
        if not cands or not remcols:
            break
        locj = random.choice(cands)
        col = random.choice(remcols)
        remcols.remove(col)
        fill(gi, col, random.randint(0, h - oh), locj)
        fill(go, col, loci, locj)
        itv -= set(range(locj, locj + ow))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry, which rotates each column of an object's
    band by the same colour-determined offset): every non-background COLOUR is one
    rigid unit living in its own column band; it slides vertically so that its top
    row lands on the top row of the colour-1 anchor (same shape everywhere, so top
    alignment == bottom alignment). The unit is selected as "all cells of that
    colour", the offset is measured from top rows, all from I alone.
    A 0-coloured unit cannot be grabbed by ARCLE Move (it grabs nonzero cells only),
    so that one is painted at its destination and its vacated cells restored to bgc.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: canvas colour; consistent across demonstrations, majority in I
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    colours = sorted(set(int(v) for v in I.flatten().tolist()) - {bgc})
    units = {}
    for col in colours:
        units[col] = [(int(r), int(c)) for r, c in zip(*np.where(I == col))]

    if 1 not in units:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    anchor_top = min(r for r, _ in units[1])

    # left-to-right over column bands
    for col in sorted(units, key=lambda k: min(c for _, c in units[k])):
        if col == 1:
            continue
        cells = units[col]
        dr = anchor_top - min(r for r, _ in cells)
        if dr == 0:
            continue
        dst = [(r + dr, c) for r, c in cells]
        if col == 0:
            ops.append(0); sels.append(sel_of(dst))
            hole = sorted(set(cells) - set(dst))
            if hole:
                ops.append(bgc); sels.append(sel_of(hole))
            continue
        move_op = 21 if dr > 0 else 20
        step = 1 if dr > 0 else -1
        cur = list(cells)
        ops.append(move_op); sels.append(sel_of(cur))          # grab colour unit
        cur = [(r + step, c) for r, c in cur]
        for _ in range(abs(dr) - 1):
            ops.append(move_op); sels.append(sel_of([]))       # keep same grab
            cur = [(r + step, c) for r, c in cur]
        hole = sorted(set(cells) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))

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
                        f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1caeab9d"
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
                                f"for task 1caeab9d"
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
                    f"Failed to build a complete episode for task 1caeab9d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1caeab9d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
