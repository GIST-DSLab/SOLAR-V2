"""
ARC Task: ea32f347 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # 1, 2, 4 are the output colors -> generator never uses them as input colors
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    return {
        "bgc": bgc,
        "col_long": random.choice(remcols),
        "col_mid": random.choice(remcols),
        "col_short": random.choice(remcols),
    }


def generate(diff_lb, diff_ub, max_h, max_w, bgc, col_long, col_mid, col_short) -> dict:
    def unifint(lo, hi):
        lo = int(lo)
        hi = int(hi)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    h = unifint(5, max(5, max_h))
    w = unifint(5, max(5, max_w))
    m = min(h, w)

    # strictly decreasing lengths -> largest / smallest are unambiguous
    a = unifint(3, min(30, m))
    b = unifint(2, a - 1)
    c = unifint(1, b - 1)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    blocked = set()  # placed cells plus their direct neighbours

    for col, l, ocol in ((col_long, a, 1), (col_mid, b, 4), (col_short, c, 2)):
        placed = False
        for _ in range(4000):
            orients = []
            if w >= l:
                orients.append("h")
            if h >= l:
                orients.append("v")
            if not orients:
                raise Exception("line does not fit")
            o = random.choice(orients)
            if o == "h":
                r = random.randint(0, h - 1)
                c0 = random.randint(0, w - l)
                cells = [(r, c0 + k) for k in range(l)]
            else:
                r0 = random.randint(0, h - l)
                c0 = random.randint(0, w - 1)
                cells = [(r0 + k, c0) for k in range(l)]
            if any(p in blocked for p in cells):
                continue
            for (rr, cc) in cells:
                gi[rr][cc] = col
                go[rr][cc] = ocol
            for (rr, cc) in cells:
                blocked.add((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    blocked.add((rr + dr, cc + dc))
            placed = True
            break
        if not placed:
            raise Exception("could not place line")

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the canvas colour the generator paints before drawing lines
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- measure the objects from I (never from O) ---
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = I[r, c]
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            objs.append(comp)

    # rule measured from I: longest line -> 1, shortest line -> 2, everything else -> 4
    order = sorted(range(len(objs)), key=lambda i: len(objs[i]), reverse=True)
    target = {}
    if order:
        target[order[0]] = 1          # largest
        target[order[-1]] = 2         # smallest
        for i in order[1:-1]:
            target[i] = 4             # middle ones

    # emit longest first, then middles, then shortest (rule order, whole objects)
    for i in order:
        comp = objs[i]
        new_col = target[i]
        if I[comp[0]] == new_col:
            continue
        sr, sc = comp[0]
        # each line is one connected same-colour region -> FloodFill from a single seed
        ops.append(10 + new_col)
        sels.append(sel_of([(sr, sc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task ea32f347"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ea32f347"
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
                                f"for task ea32f347"
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
                    f"Failed to build a complete episode for task ea32f347 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ea32f347-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
