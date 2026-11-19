"""
ARC Task: 253bf280 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # 3 is the reserved "connector" colour — it can be neither background nor foreground.
    colopts = [c for c in range(10) if c != 3]
    bgc = random.choice(colopts)
    fgc = random.choice([c for c in colopts if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    s = None
    gi = None
    for _ in range(30):
        num = unifint(diff_lb, diff_ub, card_bounds)
        num = min(max(num, 2), len(inds))
        cand_s = sample(inds, num)
        cand = fill(c, fgc, cand_s)
        # keep only scatterings that actually produce at least one connector line
        ok = any(r.count(fgc) > 1 for r in cand) or any(r.count(fgc) > 1 for r in dmirror(cand))
        if ok:
            s, gi = cand_s, cand
            break
    if gi is None:
        s = frozenset({(0, 0), (0, w - 1)})
        gi = fill(c, fgc, s)

    resh = frozenset()
    for x, r in enumerate(gi):
        if r.count(fgc) > 1:
            resh = combine(resh, connect((x, r.index(fgc)), (x, -1 + w - r[::-1].index(fgc))))
    go = fill(c, 3, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgc) > 1:
            resv = combine(resv, connect((x, r.index(fgc)), (x, -1 + h - r[::-1].index(fgc))))
    go = dmirror(fill(dmirror(go), 3, resv))
    go = fill(go, fgc, s)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule: every row (column) holding two or more foreground dots gets one straight
    connector drawn between its outermost dots, in colour 3; the dots themselves stay.

    The account below is per-LINE, not per-cell: one Color3 op draws one whole connector
    (a contiguous 1xk / kx1 run, exactly the object the rule creates), painted right over
    the dots it spans, and a single final Color<fgc> op lays the dots back on top of the
    lines that were drawn through them.
    """
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    full = [0, 0, h - 1, w - 1]  # whole grid: selection IS exactly this rectangle

    cnt = Counter(I.flatten().tolist())
    if len(cnt) < 2:                       # nothing but background: no dots, no lines
        ops.append(34); sels.append(full)
        return ops, sels
    bgc = cnt.most_common(1)[0][0]         # generator paints the canvas bgc, dots cover <= 1/4
    fgc = min((c for c in cnt if c != bgc), key=lambda c: (cnt[c], c))

    # --- the connectors, measured from I alone -------------------------------
    lines = []   # (cells of the connector, cells of it that must end up 3)
    for r in range(h):
        cs = [c for c in range(w) if I[r, c] == fgc]
        if len(cs) > 1:
            cells = [(r, c) for c in range(cs[0], cs[-1] + 1)]
            lines.append((cells, frozenset(p for p in cells if I[p] != fgc)))
    for c in range(w):
        rs = [r for r in range(h) if I[r, c] == fgc]
        if len(rs) > 1:
            cells = [(r, c) for r in range(rs[0], rs[-1] + 1)]
            lines.append((cells, frozenset(p for p in cells if I[p] != fgc)))

    # Drop connectors that leave no trace: ones made only of dots (adjacent dots), and
    # ones whose every non-dot cell is already claimed by other connectors.
    kept = [i for i, (cells, need) in enumerate(lines) if need]
    changed = True
    while changed:
        changed = False
        for i in kept:
            others = set()
            for j in kept:
                if j != i:
                    others |= lines[j][1]
            if lines[i][1] <= others:
                kept.remove(i)
                changed = True
                break

    # --- draw each connector whole, one op per line --------------------------
    grid = I.copy()
    for i in kept:
        cells, need = lines[i]
        if all(grid[p] == 3 for p in need):
            continue
        ops.append(3); sels.append(sel_of(cells))
        for p in cells:
            grid[p] = 3

    # --- lay the dots back on top of the lines drawn through them ------------
    restore = [(r, c) for r in range(h) for c in range(w)
               if I[r, c] == fgc and grid[r, c] == 3]
    if restore:
        ops.append(int(fgc)); sels.append(sel_of(restore))

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 253bf280"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 253bf280"
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
                                f"for task 253bf280"
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
                    f"Failed to build a complete episode for task 253bf280 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"253bf280-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
