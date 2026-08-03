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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (0, 8)]      # bgc=0 and 8 are reserved
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             barcol=None, cola=None, colb=None, transpose=None) -> dict:
    if transpose is None:
        transpose = choice((True, False))
    if barcol is None:
        barcol = choice(remove(0, remove(8, interval(0, 10, 1))))
    if cola is None:
        cola = choice(remove(barcol, remove(0, remove(8, interval(0, 10, 1)))))
    if colb is None:
        colb = choice(remove(barcol, remove(0, remove(8, interval(0, 10, 1)))))

    # input is (h, 2w+1); dmirror swaps the two axes
    if transpose:
        h_ub = min(30, max_w)
        w_ub = min(14, (max_h - 1) // 2)
    else:
        h_ub = min(30, max_h)
        w_ub = min(14, (max_w - 1) // 2)
    h_ub = max(2, h_ub)
    w_ub = max(2, w_ub)

    h = unifint(diff_lb, diff_ub, (2, h_ub))
    w = unifint(diff_lb, diff_ub, (2, w_ub))

    canv = canvas(0, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 8, ofcolor(gia, 0) & ofcolor(gib, 0))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # The bar (a full line of one color) splits I into two equal halves;
    # the output has the shape of one half, so the halves lie side by side
    # when the heights match, and stacked otherwise.  Half A is always at (0,0).
    vertical = (hi == ho)
    A = I[:ho, :wo]
    if vertical:
        br, bc = 0, wo + 1
    else:
        br, bc = ho + 1, 0
    B = I[br:br + ho, bc:bc + wo]

    def regions(mask, colors=None):
        """4-connected components of `mask`; if `colors` given, split by color too."""
        seen = np.zeros(mask.shape, dtype=bool)
        out = []
        for r in range(mask.shape[0]):
            for c in range(mask.shape[1]):
                if not mask[r, c] or seen[r, c]:
                    continue
                col = None if colors is None else colors[r, c]
                comp = []
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] \
                                and mask[ny, nx] and not seen[ny, nx] \
                                and (col is None or colors[ny, nx] == col):
                            seen[ny, nx] = True
                            q.append((ny, nx))
                out.append(comp)
        return out

    G = I.copy()
    ops, sels = [], []

    # 1) Paint the empty area of half A with 8 -- one fill per empty region.
    #    Skip regions fully covered by B's marks: they can never survive step 2.
    empty = [reg for reg in regions(A == 0) if any(B[r, c] == 0 for r, c in reg)]
    empty.sort(key=lambda reg: (-len(reg), reg[0]))
    for reg in empty:
        r, c = reg[0]
        ops.append(18)
        sels.append([r, c, 0, 0])
        for (y, x) in reg:
            G[y, x] = 8

    # 2) Lay half B on top of half A: every mark of B covers an 8, so only the
    #    cells empty in BOTH halves stay 8.
    if bool(np.any((A == 0) & (B != 0))):
        ops.append(28)
        sels.append([br, bc, ho - 1, wo - 1])
        ops.append(30)
        sels.append([0, 0, 0, 0])
        m = B != 0
        G[:ho, :wo][m] = B[m]

    # 3) Keep only half A (it now carries the answer, no cell is 0 anymore).
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])
    G = G[:ho, :wo].copy()

    # 4) Erase the leftover marks of both halves, region by region.
    marks = regions(G != 8, colors=G)
    marks.sort(key=lambda reg: (-len(reg), reg[0]))
    for reg in marks:
        r, c = reg[0]
        ops.append(10)
        sels.append([r, c, 0, 0])
        for (y, x) in reg:
            G[y, x] = 0

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
