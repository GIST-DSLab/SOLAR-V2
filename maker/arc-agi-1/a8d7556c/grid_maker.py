"""
ARC Task: a8d7556c (RE-ARC) — LLM-generated grid_maker
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
    # generator: cols = interval(0,10) minus (0, 2); fgc = choice(cols)
    cols = [c for c in range(10) if c not in (0, 2)]
    fgc = random.choice(cols)
    return {"fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    c = canvas(fgc, (h, w))
    numblacks = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = totuple(asindices(c))
    blacks = sample(inds, numblacks)
    gi = fill(c, 0, blacks)
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    sqlocs = sample(inds, numsq)
    gi = fill(gi, 0, shift(sqlocs, (0, 0)))
    gi = fill(gi, 0, shift(sqlocs, (0, 1)))
    gi = fill(gi, 0, shift(sqlocs, (1, 0)))
    gi = fill(gi, 0, shift(sqlocs, (1, 1)))
    go = tuple(e for e in gi)
    for a in range(h - 1):
        for b in range(w - 1):
            if gi[a][b] == 0 and gi[a + 1][b] == 0 and gi[a][b + 1] == 0 and gi[a + 1][b + 1] == 0:
                go = fill(go, 2, {(a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)})
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    ops, sels = [], []

    # --- Rule measured from I only: every 2x2 block that is entirely 0 gets stamped with color 2.
    occ = []
    for a in range(hi - 1):
        for b in range(wi - 1):
            if I[a, b] == 0 and I[a + 1, b] == 0 and I[a, b + 1] == 0 and I[a + 1, b + 1] == 0:
                occ.append((a, b))

    def cells_of(o):
        a, b = o
        return frozenset([(a, b), (a + 1, b), (a, b + 1), (a + 1, b + 1)])

    union = set()
    for o in occ:
        union |= cells_of(o)

    # --- Group occurrences by the connected stamped region they belong to (legible ordering)
    comp_id = {}
    comps = []
    for cell in sorted(union):
        if cell in comp_id:
            continue
        idx = len(comps)
        comp = []
        dq = deque([cell])
        comp_id[cell] = idx
        while dq:
            r, c = dq.popleft()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if nb in union and nb not in comp_id:
                    comp_id[nb] = idx
                    dq.append(nb)
        comps.append(sorted(comp))

    ordered = sorted(occ, key=lambda o: (comp_id[(o[0], o[1])], o[0], o[1]))

    # --- Keep only stamps that actually contribute new cells (no op whose effect is fully
    #     covered by the others: overlapping occurrences would be invisible duplicates)
    kept = []
    covered = set()
    for o in ordered:
        cs = cells_of(o)
        if not cs <= covered:
            kept.append(o)
            covered |= cs

    changed = True
    while changed:
        changed = False
        for i in range(len(kept) - 1, -1, -1):
            others = set()
            for j, o in enumerate(kept):
                if j != i:
                    others |= cells_of(o)
            if cells_of(kept[i]) <= others:
                kept.pop(i)
                changed = True
                break

    # --- Stamp the 2x2 unit at each retained occurrence
    for o in kept:
        ops.append(2)
        sels.append(sel_of(sorted(cells_of(o))))

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
                        f"num_examples+1 ({num_examples + 1}) for task a8d7556c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a8d7556c"
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
                                f"for task a8d7556c"
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
                    f"Failed to build a complete episode for task a8d7556c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a8d7556c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
