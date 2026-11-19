"""
ARC Task: 363442ee (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    barcol = random.choice(rem)
    rem = [c for c in rem if c != barcol]
    dotcol = random.choice(rem)
    return {"bgc": bgc, "barcol": barcol, "dotcol": dotcol}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, barcol, dotcol) -> dict:
    import numpy as np

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi_ = a + round((b - a) * ub)
        lo, hi_ = min(lo, hi_), max(lo, hi_)
        return random.randint(lo, hi_)

    m = min(max_h, max_w)
    h = unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    w = unifint(diff_lb, diff_ub, (1, 3)) * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, max(2, m // h)))
    nremw = unifint(diff_lb, diff_ub, (2, max(2, (m - w - 1) // w)))
    rsh, rsw = nremh * h, nremw * w
    remcols = [c for c in range(10) if c not in (bgc, barcol)]
    rsi = np.full((rsh, rsw), bgc, dtype=int)
    rso = rsi.copy()
    ls = np.full(((nremh - 1) * h, w), bgc, dtype=int)
    bar = np.full((rsh, 1), barcol, dtype=int)
    dotcands = [(i, j) for i in range(0, rsh, h) for j in range(0, rsw, w)]
    dev = unifint(diff_lb, diff_ub, (1, max(1, len(dotcands) // 2)))
    ndots = random.choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = random.sample(dotcands, ndots)
    nf = unifint(diff_lb, diff_ub, (1, len(remcols)))
    fullremcols = random.sample(remcols, nf)
    ulc = np.array([[random.choice(fullremcols) for _ in range(w)] for _ in range(h)], dtype=int)
    for (i, j) in dots:
        rsi[i + h // 2, j + w // 2] = dotcol
        rso[i:i + h, j:j + w] = ulc
    left = np.vstack([ulc, ls])
    gi = np.hstack([left, bar, rsi])
    go = np.hstack([left, bar, rso])
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi], "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # find the bar: a full uniform row/column of a non-background colour
    bar = None
    for r in range(H):
        if len(set(I[r].tolist())) == 1 and I[r, 0] != bgc:
            bar = ("row", r)
    for c in range(W):
        if len(set(I[:, c].tolist())) == 1 and I[0, c] != bgc:
            bar = ("col", c)
    kind, k = bar
    if kind == "row":
        sides = [((0, k), (0, W)), ((k + 1, H), (0, W))]
        sides.sort(key=lambda s: s[0][1] - s[0][0])
    else:
        sides = [((0, H), (0, k)), ((0, H), (k + 1, W))]
        sides.sort(key=lambda s: s[1][1] - s[1][0])
    # narrower side holds the template tile, wider side holds the dots
    (tr0, tr1), (tc0, tc1) = sides[0]
    (rr0, rr1), (rc0, rc1) = sides[1]
    tcells = [(r, c) for r in range(tr0, tr1) for c in range(tc0, tc1) if I[r, c] != bgc]
    r0 = min(r for r, _ in tcells); r1 = max(r for r, _ in tcells)
    c0 = min(c for _, c in tcells); c1 = max(c for _, c in tcells)
    th, tw = r1 - r0 + 1, c1 - c0 + 1
    tile = I[r0:r1 + 1, c0:c1 + 1]
    zeros = [(i, j) for i in range(th) for j in range(tw) if tile[i, j] == 0]
    has_nz = bool((tile != 0).any())
    dots = [(r, c) for r in range(rr0, rr1) for c in range(rc0, rc1) if I[r, c] != bgc]
    ops, sels = [], []
    if has_nz and dots:
        # copy the whole template rectangle (every cell is template content)
        ops.append(28); sels.append([r0, c0, th - 1, tw - 1])
    for (dr, dc) in dots:
        tr, tc = dr - th // 2, dc - tw // 2
        if has_nz:
            ops.append(30); sels.append([tr, tc, 0, 0])
        # template's 0-cells are not carried by Paste; paint those still non-0 now
        zc = [(tr + i, tc + j) for i, j in zeros if I[tr + i, tc + j] != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))
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
                        f"num_examples+1 ({num_examples + 1}) for task 363442ee"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 363442ee"
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
                                f"for task 363442ee"
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
                    f"Failed to build a complete episode for task 363442ee "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"363442ee-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
