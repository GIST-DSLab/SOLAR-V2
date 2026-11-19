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
import numpy as np
from collections import Counter

# Note: the code-golf entry reads this task the same way the maker does.
# It slides a window, counts the 1s in each window and returns the window
# with the most 1s. Its fixed 3x3 window is only a golf shortcut. Here the
# "window" is each placed rectangle, found as a connected non-background
# component. Both readings use the same unit (one rectangle), the same
# quantity (how many 1s it holds) and the same action (crop to the winner).
# Nothing in the operations differs, so the crop-to-winner route is kept.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])

    H = max(6, min(30, max_h))
    W = max(6, min(30, max_w))
    for _attempt in range(100):
        h = unifint(diff_lb, diff_ub, (6, H))
        w = unifint(diff_lb, diff_ub, (6, W))
        dh = unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
        dw = unifint(diff_lb, diff_ub, (2, max(2, w // 3)))
        num = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * dh * dw))))
        cards = list(range(dh * dw))
        ccards = sorted(random.sample(cards, min(num, len(cards))))
        inds = [(i, j) for i in range(dh) for j in range(dw)]
        sgs = []
        for card in ccards:
            x = set(random.sample(inds, card))
            sg = [((i, j), 1 if (i, j) in x else fgc) for (i, j) in inds]
            sgs.append(sg)
        go = np.full((dh, dw), fgc, dtype=int)
        for (i, j), v in sgs[-1]:
            go[i, j] = v
        gi = np.full((h, w), bgc, dtype=int)
        inds2 = set((i, j) for i in range(h - dh) for j in range(w - dw))
        maxtr = 10
        placed_winner = False
        for k, sg in enumerate(sgs[::-1]):
            if len(inds2) == 0:
                break
            tr = 0
            loc = random.choice(sorted(inds2))
            cells = [(loc[0] + i, loc[1] + j) for (i, j), _ in sg]
            while (not set(cells).issubset(inds2)) and tr < maxtr:
                loc = random.choice(sorted(inds2))
                cells = [(loc[0] + i, loc[1] + j) for (i, j), _ in sg]
                tr += 1
            if tr < maxtr:
                r0, c0 = loc
                r1, c1 = r0 + dh - 1, c0 + dw - 1
                ob = set()
                for r in range(r0 - 1, r1 + 2):
                    for c in range(c0 - 1, c1 + 2):
                        ob.add((r, c))
                inds2 -= ob
                for (i, j), v in sg:
                    gi[r0 + i, c0 + j] = v
                if k == 0:
                    placed_winner = True
        if not placed_winner:
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: one fixed canvas colour per episode. Take the colour that is
    # most common across this input and all demonstration inputs.
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            try:
                ei = ex[0] if not isinstance(ex, dict) else ex["input"]
                cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
            except Exception:
                pass
    bgc = cnt.most_common(1)[0][0]

    # Each placed rectangle is one 4-connected non-background component
    # (rectangles are kept apart by their outbox). The winner is the
    # rectangle holding the most colour-1 cells.
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
                        if 0 <= ny < hi and 0 <= nx < wi and \
                           I[ny, nx] != bgc and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                k = sum(1 for (y, x) in cells if I[y, x] == 1)
                if k > best_count:
                    best_count, best_cells = k, cells

    rs = [y for y, x in best_cells]
    cs = [x for y, x in best_cells]
    r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # The winner is a full solid rectangle, so its bbox is exactly its cells.
    # Crop the canvas to it. If fgc is 0, those cells are cleared to 0 by the
    # crop anyway, so the result is still correct.
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
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
