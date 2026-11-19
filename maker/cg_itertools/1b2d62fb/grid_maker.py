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

VARIANTS = [{"transpose": False}, {"transpose": True}]

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (0, 8)]
    barcol = random.choice(cols)
    rem = [c for c in cols if c != barcol]
    cola = random.choice(rem)
    colb = random.choice(rem)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"barcol": barcol, "cola": cola, "colb": colb, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h, max_w, barcol=None, cola=None, colb=None,
             transpose=None, **kw) -> dict:
    def unifint(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))
    if transpose is None:
        transpose = random.choice((True, False))
    if barcol is None:
        c = sample_colors(1)
        barcol, cola, colb = c["barcol"], c["cola"], c["colb"]
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(2, min(30, lim_h))
    w = unifint(2, min(14, (lim_w - 1) // 2))
    mp = (h * w) // 2
    deva = unifint(0, mp) * random.choice((1, -1))
    devb = unifint(0, mp) * random.choice((1, -1))
    numa = max(min(h * w - 1, mp + deva), 1)
    numb = max(min(h * w - 1, mp + devb), 1)
    inds = [(i, j) for i in range(h) for j in range(w)]
    ga = np.zeros((h, w), dtype=int)
    gb = np.zeros((h, w), dtype=int)
    for (i, j) in random.sample(inds, numa):
        ga[i, j] = cola
    for (i, j) in random.sample(inds, numb):
        gb[i, j] = colb
    gi = np.hstack([ga, np.full((h, 1), barcol), gb])
    go = np.where((ga == 0) & (gb == 0), 8, 0)
    if transpose:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The bar is a full line of one colour through the middle: if the middle
    # column is monochrome the halves sit side by side, else they are stacked.
    mid = I[:, wi // 2] if wi % 2 == 1 else None
    vertical = mid is not None and len(set(mid.tolist())) == 1 and mid[0] != 0
    if vertical:
        ho, wo = hi, wi // 2
        br, bc = 0, wo + 1
    else:
        ho, wo = hi // 2, wi
        br, bc = ho + 1, 0
    A = I[:ho, :wo]
    B = I[br:br + ho, bc:bc + wo]
    ops, sels = [], []
    G = I.copy()
    # 1) Overlay the second half on the first: union of both halves' marks
    #    (a cell stays empty only if empty in both halves).
    if np.any((A == 0) & (B != 0)):
        ops.append(28); sels.append([br, bc, ho - 1, wo - 1])  # whole half-B rectangle
        ops.append(30); sels.append([0, 0, 0, 0])
        G[:ho, :wo][B != 0] = B[B != 0]
    # 2) Keep only the overlaid half.
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # whole half-A rectangle
    G = G[:ho, :wo].copy()
    # 3) Complement: each region still empty after the overlay is filled with 8 ...
    seen = np.zeros((ho, wo), dtype=bool)
    regs = []
    for r in range(ho):
        for c in range(wo):
            if G[r, c] != 0 or seen[r, c]:
                continue
            comp, st = [], [(r, c)]
            seen[r, c] = True
            while st:
                y, x = st.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < ho and 0 <= nx < wo and not seen[ny, nx] and G[ny, nx] == 0:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            regs.append(sorted(comp))
    regs.sort(key=lambda g: (-len(g), g[0]))
    for reg in regs:
        r, c = reg[0]
        ops.append(18); sels.append([r, c, 0, 0])
        for y, x in reg:
            G[y, x] = 8
    # ... and every mark of the union is wiped back to 0, one same-colour patch at a time.
    seen = np.zeros((ho, wo), dtype=bool)
    marks = []
    for r in range(ho):
        for c in range(wo):
            if G[r, c] == 8 or seen[r, c]:
                continue
            col = G[r, c]
            comp, st = [], [(r, c)]
            seen[r, c] = True
            while st:
                y, x = st.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < ho and 0 <= nx < wo and not seen[ny, nx] and G[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            marks.append(sorted(comp))
    marks.sort(key=lambda g: (-len(g), g[0]))
    for reg in marks:
        # seed on a mark that half A already had, when the patch contains one
        own = [p for p in reg if A[p] != 0]
        r, c = (own or reg)[0]
        ops.append(10); sels.append([r, c, 0, 0])
        for y, x in reg:
            G[y, x] = 0
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
