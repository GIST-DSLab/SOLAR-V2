"""
ARC Task: 1b2d62fb (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(1, 10) if c != 8]
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barcol=None, cola=None, colb=None, mirrored=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))
    cols = [c for c in range(1, 10) if c != 8]
    if barcol is None:
        barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    if cola is None:
        cola = random.choice(rem)
    if colb is None:
        colb = random.choice(rem)
    if mirrored is None:
        mirrored = random.choice((True, False))
    lim = min(max_h, max_w, 30)
    h = unifint(diff_lb, diff_ub, (2, lim))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(14, (lim - 1) // 2))))
    inds = [(i, j) for i in range(h) for j in range(w)]
    mp = (h * w) // 2

    def num():
        d = unifint(diff_lb, diff_ub, (0, mp)) * random.choice((1, -1))
        return max(min(h * w - 1, mp + d), 1)
    a = random.sample(inds, num())
    b = random.sample(inds, num())
    gia = np.zeros((h, w), dtype=int)
    gib = np.zeros((h, w), dtype=int)
    for p in a:
        gia[p] = cola
    for p in b:
        gib[p] = colb
    bar = np.full((h, 1), barcol, dtype=int)
    gi = np.hstack([gia, bar, gib])
    go = np.where((gia == 0) & (gib == 0), 8, 0)
    if mirrored:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the two halves either side of the bar are paired cell-for-cell;
    # a cell is 8 exactly when it is empty in BOTH halves.  So: keep half A,
    # overlay half B onto it (union of marks), mark the still-empty regions 8,
    # then wipe out every mark.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Locate the bar from I (as the rule does): is the middle column one colour?
    midcol = I[:, wi // 2]
    vertical = len(set(midcol.tolist())) == 1 and wi % 2 == 1
    if vertical:
        ho, wo = hi, wi // 2
        br, bc = 0, wo + 1
    else:
        ho, wo = hi // 2, wi
        br, bc = ho + 1, 0
    B = I[br:br + ho, bc:bc + wo]

    def regions(G, pred):
        seen = np.zeros(G.shape, dtype=bool)
        out = []
        for r in range(G.shape[0]):
            for c in range(G.shape[1]):
                if seen[r, c] or not pred(G[r, c]):
                    continue
                col = G[r, c]
                comp = []
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < G.shape[0] and 0 <= nx < G.shape[1] \
                                and not seen[ny, nx] and G[ny, nx] == col:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(comp)
        out.sort(key=lambda reg: (-len(reg), reg[0]))
        return out

    ops, sels = [], []
    # 1) Keep the first half as the canvas (full half rectangle, crop).
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])
    G = I[:ho, :wo].copy()
    # 2) Lay the second half (full rectangle of I) over it cell-for-cell.
    if np.any((B != 0) & (G != B)):
        ops.append(28); sels.append([br, bc, ho - 1, wo - 1])
        ops.append(30); sels.append(sel_of([(0, 0)]))
        G[B != 0] = B[B != 0]
    # 3) Cells empty in both halves: flood each empty region with 8.
    for reg in regions(G, lambda v: v == 0):
        r, c = reg[0]
        ops.append(18); sels.append(sel_of([(r, c)]))
        for p in reg:
            G[p] = 8
    # 4) Every mark from either half becomes empty, region by region.
    for reg in regions(G, lambda v: v != 8):
        r, c = reg[0]
        ops.append(10); sels.append(sel_of([(r, c)]))
        for p in reg:
            G[p] = 0
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
                        f"num_examples+1 ({num_examples + 1}) for task 1b2d62fb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1b2d62fb"
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
                                f"for task 1b2d62fb"
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
                    f"Failed to build a complete episode for task 1b2d62fb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1b2d62fb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
