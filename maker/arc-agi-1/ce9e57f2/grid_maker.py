"""
ARC Task: ce9e57f2 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random
from collections import Counter, deque


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]  # 8 reserved as marker
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        lo = int(round(lo)); hi = int(round(hi))
        lo = max(a, min(b, lo)); hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c != 8]

    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    nbars = unifint(diff_lb, diff_ub, (2, (w - 2) // 2))

    locopts = list(range(1, w - 1))
    barlocs = []
    for _ in range(nbars):
        if not locopts:
            break
        loc = random.choice(locopts)
        barlocs.append(loc)
        for x in (loc, loc + 1, loc - 1):
            if x in locopts:
                locopts.remove(x)

    remcols = [c for c in cols if c != bgc]
    numc = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    colss = random.sample(remcols, numc)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)

    for j in barlocs:
        barloci = unifint(diff_lb, diff_ub, (1, h - 2))
        half_last = barloci // 2 if barloci % 2 == 1 else (barloci - 1) // 2
        barcol = random.choice(colss)
        for r in range(0, barloci + 1):
            gi[r, j] = barcol
            go[r, j] = barcol
        for r in range(0, half_last + 1):
            go[r, j] = 8

    # rotation variant: choose k that still fits max dims after rotation
    allowed = [0, 2]
    if h <= max_w and w <= max_h:
        allowed += [1, 3]
    k = random.choice(allowed)
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    visited = np.zeros((hi, wi), dtype=bool)

    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or visited[r, c]:
                continue
            col = I[r, c]
            comp = []
            dq = deque([(r, c)])
            visited[r, c] = True
            while dq:
                y, x = dq.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not visited[ny, nx] and I[ny, nx] == col:
                        visited[ny, nx] = True
                        dq.append((ny, nx))

            rows = [p[0] for p in comp]
            colsc = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(colsc), max(colsc)

            if c0 == c1:  # vertical bar
                L = r1 - r0 + 1
                half = L // 2
                if half <= 0:
                    continue
                if r0 == 0:            # anchored at top edge
                    sels.append([0, c0, half - 1, 0])
                else:                  # anchored at bottom edge
                    sels.append([r1 - half + 1, c0, half - 1, 0])
                ops.append(8)
            else:  # horizontal bar
                L = c1 - c0 + 1
                half = L // 2
                if half <= 0:
                    continue
                if c0 == 0:            # anchored at left edge
                    sels.append([r0, 0, 0, half - 1])
                else:                  # anchored at right edge
                    sels.append([r0, c1 - half + 1, 0, half - 1])
                ops.append(8)

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ce9e57f2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ce9e57f2"
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
                                f"for task ce9e57f2"
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
                    f"Failed to build a complete episode for task ce9e57f2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ce9e57f2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
