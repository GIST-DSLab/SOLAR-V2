"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of clockwise quarter turns applied to the generated pair


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # fgc is kept nonzero so the cross lines can be grabbed and slid with Move
    # (ARCLE never moves 0-valued cells)
    fgc = random.choice([c for c in cols if c != bgc and c != 0])
    dotc = random.choice([c for c in cols if c not in (bgc, fgc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, dotc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    if rot is None:
        rot = random.choice(ROTS)
    # a quarter turn swaps h and w, so the limits swap with it
    lim_h, lim_w = (max_h, max_w) if rot % 2 == 0 else (max_w, max_h)
    h = max(3, min(lim_h, unifint(diff_lb, diff_ub, (3, max(3, lim_h)))))
    w = max(3, min(lim_w, unifint(diff_lb, diff_ub, (3, max(3, lim_w)))))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = [(i, j) for i in range(loci) for j in range(locj)]
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = max(1, min(maxn, unifint(diff_lb, diff_ub, (1, maxn))))
    ss = random.sample(inds, nn)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, :] = fgc
    gi[:, locj] = fgc
    for (i, j) in ss:
        gi[i, j] = dotc
    go[loci + nn, :] = fgc
    go[:, locj + nn] = fgc
    gi = np.rot90(gi, k=-rot)
    go = np.rot90(go, k=-rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # The cross: the only uniform full row and the only uniform full column
    old_hrow = next(r for r in range(hi) if np.all(I[r, :] == I[r, 0]))
    fgc = int(I[old_hrow, 0])
    old_vcol = next(c for c in range(wi) if np.all(I[:, c] == I[0, c]))

    # The two other colours: background (more common) and the dots (rarer)
    cnt = Counter(int(v) for v in I.flatten() if v != fgc)
    rest = cnt.most_common()
    bgc = rest[0][0]
    dotc = rest[1][0]
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    nn = len(dots)

    # The cross moves nn cells away from the dots, on both axes
    mean_r = sum(r for r, _ in dots) / nn
    mean_c = sum(c for _, c in dots) / nn
    di = sign(old_hrow - mean_r)
    dj = sign(old_vcol - mean_c)
    new_hrow = old_hrow + di * nn
    new_vcol = old_vcol + dj * nn

    ops, sels = [], []

    # 1. Paint all the dots (every cell of the dot colour) to the background
    ops.append(int(bgc)); sels.append(sel_of(dots))

    # 2. Slide the column arm sideways nn cells. The arm is the column without
    #    the cell it shares with the row, which stays in place as part of the row.
    col_cells = [(r, old_vcol) for r in range(hi) if r != old_hrow]
    move_c = 22 if dj > 0 else 23
    ops.append(move_c); sels.append(sel_of(col_cells))
    for _ in range(nn - 1):
        ops.append(move_c); sels.append(sel_of([]))
    if bgc != 0:
        # only the column's starting cells are left at 0; repaint them to bgc
        ops.append(int(bgc)); sels.append(sel_of(col_cells))

    # 3. Slide the row arm up or down nn cells, leaving out the cell where it
    #    now crosses the moved column
    row_cells = [(old_hrow, c) for c in range(wi) if c != new_vcol]
    move_r = 21 if di > 0 else 20
    ops.append(move_r); sels.append(sel_of(row_cells))
    for _ in range(nn - 1):
        ops.append(move_r); sels.append(sel_of([]))
    if bgc != 0:
        ops.append(int(bgc)); sels.append(sel_of(row_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
