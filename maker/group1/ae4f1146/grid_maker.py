"""
ARC Task: ae4f1146 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi_ = a + int(round(ub * (b - a)))
        if hi_ < lo:
            hi_ = lo
        return random.randint(lo, hi_)

    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])

    for _ in range(100):
        h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
        w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
        dh = unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
        dw = unifint(diff_lb, diff_ub, (2, max(2, w // 3)))
        num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * dh * dw))))
        cards = list(range(dh * dw))
        ccards = sorted(random.sample(cards, min(num, len(cards))))
        inds = [(r, c) for r in range(dh) for c in range(dw)]
        sgs = []
        for card in ccards:
            g = [[fgc] * dw for _ in range(dh)]
            for (r, c) in random.sample(inds, card):
                g[r][c] = 1
            sgs.append(g)
        go = [row[:] for row in sgs[-1]]
        gi = [[bgc] * w for _ in range(h)]
        inds2 = {(r, c) for r in range(h - dh) for c in range(w - dw)}
        placed_last = False
        for k, sg in enumerate(sgs[::-1]):
            if not inds2:
                break
            ok = False
            for _t in range(10):
                loc = random.choice(sorted(inds2))
                cells = {(loc[0] + r, loc[1] + c) for r in range(dh) for c in range(dw)}
                if cells.issubset(inds2):
                    ok = True
                    break
            if ok:
                r0, c0 = loc
                outb = {(r, c) for r in range(r0 - 1, r0 + dh + 1)
                        for c in range(c0 - 1, c0 + dw + 1)}
                inds2 -= outb
                for r in range(dh):
                    for c in range(dw):
                        gi[r0 + r][c0 + c] = sg[r][c]
                if k == 0:
                    placed_last = True
        if placed_last:
            return {"input": gi, "output": go}
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # components of non-background cells = the placed cards
    visited = np.zeros((hi, wi), bool)
    best_cells, best_count = None, -1
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] != bgc \
                                and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                cnt = sum(1 for (y, x) in cells if I[y, x] == 1)
                if cnt > best_count:
                    best_count, best_cells = cnt, cells

    r0 = min(y for y, x in best_cells); r1 = max(y for y, x in best_cells)
    c0 = min(x for y, x in best_cells); c1 = max(x for y, x in best_cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    card = I[r0:r1 + 1, c0:c1 + 1]
    others = [int(v) for v in card.flatten() if v != 1]
    fgc = Counter(others).most_common(1)[0][0] if others else 1

    ops, sels = [], []
    # 1. copy the winning card (full rectangle, from the input) to the clipboard
    ops.append(28); sels.append([r0, c0, h - 1, w - 1])
    # 2. shrink the canvas to the card's size (top-left region of I survives transparently)
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    grid = np.where(I[:h, :w] != 0, I[:h, :w], 0).copy()
    if np.array_equal(grid, card):
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    # 3. lay the card's base colour over the whole canvas (full rectangle)
    if not np.all(grid == fgc):
        ops.append(fgc); sels.append([0, 0, h - 1, w - 1])
        grid[:, :] = fgc
    # 4. paste the card's pattern on top (transparent paste of nonzero cells)
    pasted = grid.copy()
    pasted[card != 0] = card[card != 0]
    if not np.array_equal(pasted, grid):
        ops.append(30); sels.append([0, 0, 0, 0])
        grid = pasted
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ae4f1146"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae4f1146"
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
                                f"for task ae4f1146"
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
                    f"Failed to build a complete episode for task ae4f1146 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae4f1146-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
