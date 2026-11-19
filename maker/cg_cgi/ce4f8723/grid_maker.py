"""
ARC Task: ce4f8723 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    cola = random.choice(rem)
    colb = random.choice([c for c in rem if c != cola])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "barcol": barcol, "cola": cola, "colb": colb,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, cola, colb,
             transpose=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    if transpose is None:
        transpose = random.choice([False, True])
    # un-transposed grid is h x (2w+1)
    if transpose:
        lim_h, lim_w = max_w, max_h
    else:
        lim_h, lim_w = max_h, max_w
    h = unifint(diff_lb, diff_ub, (2, min(30, lim_h)))
    w = unifint(diff_lb, diff_ub, (2, min(14, (lim_w - 1) // 2)))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2
    deva = unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    devb = unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    a = random.sample(inds, numa)
    b = random.sample(inds, numb)
    gia = [[bgc] * w for _ in range(h)]
    gib = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for (i, j) in a:
        gia[i][j] = cola
        go[i][j] = 3
    for (i, j) in b:
        gib[i][j] = colb
        go[i][j] = 3
    gi = [gia[i] + [barcol] + gib[i] for i in range(h)]
    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: the second half is laid over the first half cell-for-cell;
    # every cell where either layer carries a mark becomes 3, cells where both
    # layers show background stay background.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    horiz = (hi % 2 == 1) and (len(set(I[hi // 2].tolist())) == 1)
    if horiz:
        k = hi // 2
        h1 = (0, 0, k, wi)            # r0, c0, h, w
        h2 = (k + 1, 0, k, wi)
    else:
        k = wi // 2
        h1 = (0, 0, hi, k)
        h2 = (0, k + 1, hi, k)
    hh, ww = h1[2], h1[3]
    half1 = I[h1[0]:h1[0] + hh, h1[1]:h1[1] + ww]
    half2 = I[h2[0]:h2[0] + hh, h2[1]:h2[1] + ww]

    pal1 = set(half1.flatten().tolist())
    pal2 = set(half2.flatten().tolist())
    common = pal1 & pal2
    if len(common) == 1:
        bgc = next(iter(common))
    else:
        from collections import Counter
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    rest1 = [c for c in pal1 if c != bgc]
    rest2 = [c for c in pal2 if c != bgc]

    G = I.copy()

    # 1) first layer: all mark-coloured cells of half1 become 3
    if rest1:
        cola = rest1[0]
        cells = [(h1[0] + r, h1[1] + c) for r in range(hh) for c in range(ww)
                 if half1[r, c] == cola]
        ops.append(3); sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = 3

    # 2) second layer: all mark-coloured cells of half2 become 3
    marks2 = []
    if rest2:
        colb = rest2[0]
        marks2 = [(r, c) for r in range(hh) for c in range(ww) if half2[r, c] == colb]
        cells = [(h2[0] + r, h2[1] + c) for (r, c) in marks2]
        ops.append(3); sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = 3

    # 3) make half2's background transparent so only its marks overlay
    if bgc != 0:
        cells = [(h2[0] + r, h2[1] + c) for r in range(hh) for c in range(ww)
                 if half2[r, c] == bgc]
        if cells:
            ops.append(0); sels.append(sel_of(cells))
            for (r, c) in cells:
                G[r, c] = 0

    # 4) lay half2 over half1: copy the whole half2 rectangle, paste at half1 origin
    if any(G[h1[0] + r, h1[1] + c] != 3 for (r, c) in marks2):
        ops.append(29); sels.append([h2[0], h2[1], hh - 1, ww - 1])  # full rectangle
        ops.append(30); sels.append([h1[0], h1[1], 0, 0])

    # 5) keep only the overlaid first half
    ops.append(33); sels.append([0, 0, hh - 1, ww - 1])
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce4f8723"
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
                                f"for task ce4f8723"
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
                    f"Failed to build a complete episode for task ce4f8723 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce4f8723-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
