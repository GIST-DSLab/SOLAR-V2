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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

ROTATIONS = [0, 1, 2, 3]  # identity, rot90, rot180, rot270


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc, dotc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTATIONS):
        examples = [{"rot": r} for r in ROTATIONS]
        examples += [{"rot": random.choice(ROTATIONS)} for _ in range(n_ex - len(ROTATIONS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTATIONS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, dotc, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        lo = max(a, min(lo, b)); hi = max(a, min(hi, b))
        return random.randint(lo, hi)

    if rot is None:
        rot = random.choice(ROTATIONS)
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = [(i, j) for i in range(loci) for j in range(locj)]
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = random.sample(inds, nn)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, :] = fgc
    gi[:, locj] = fgc
    for (i, j) in ss:
        gi[i, j] = dotc
    go[loci + nn, :] = fgc
    go[:, locj + nn] = fgc
    # rotf in DSL: rot90 = CW = np.rot90(k=3)
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # The cross: the colour owning a full row AND a full column.
    hrow = None
    for r in range(hi):
        if np.all(I[r, :] == I[r, 0]):
            hrow = r
            fgc = int(I[r, 0])
            break
    vcol = None
    for c in range(wi):
        if np.all(I[:, c] == fgc):
            vcol = c
            break

    # Of the two other colours, the background is the more common one
    # (the dots, which act as the counter, are always fewer).
    cnt = Counter(int(v) for v in I.flatten() if v != fgc)
    (bgc, _), (dotc, _) = cnt.most_common(2)
    dots = [(int(r), int(c)) for r, c in np.argwhere(I == dotc)]
    nn = len(dots)

    # The cross shifts by nn cells diagonally, away from the dots.
    mean_r = sum(r for r, _ in dots) / nn
    mean_c = sum(c for _, c in dots) / nn
    di = sign(hrow - mean_r)
    dj = sign(vcol - mean_c)
    new_hrow = hrow + di * nn
    new_vcol = vcol + dj * nn

    cross = sorted({(hrow, c) for c in range(wi)} | {(r, vcol) for r in range(hi)})
    new_row = [(new_hrow, c) for c in range(wi)]
    new_col = [(r, new_vcol) for r in range(hi)]

    ops, sels = [], []

    if fgc != 0:
        # Slide the whole cross as one object: one grab, then empty-selection moves.
        steps = [(20 if di < 0 else 21, di, 0)] * nn + [(23 if dj < 0 else 22, 0, dj)] * nn
        cur = list(cross)
        first = True
        for op, dr, dc in steps:
            ops.append(op)
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + dr, c + dc) for r, c in cur]
        cur_on = {(r, c) for r, c in cur if 0 <= r < hi and 0 <= c < wi}

        # Repair the cross's original footprint (left at 0 by the grab).
        hole = sorted(set(cross) - cur_on)
        if bgc != 0 and hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))

        # The moved lines were clipped at one edge; extend them back to full length.
        row_ext = [p for p in new_row if p not in cur_on]
        if row_ext:
            ops.append(int(fgc)); sels.append(sel_of(row_ext))
        col_ext = [p for p in new_col if p not in cur_on]
        if col_ext:
            ops.append(int(fgc)); sels.append(sel_of(col_ext))
    else:
        # fgc == 0 can't be grabbed by Move (zero cells are transparent): erase and redraw.
        ops.append(int(bgc)); sels.append(sel_of(cross))
        ops.append(int(fgc)); sels.append(sel_of(new_row))
        ops.append(int(fgc)); sels.append(sel_of(new_col))

    # The dots were only a counter; clear them all.
    ops.append(int(bgc)); sels.append(sel_of(dots))

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
