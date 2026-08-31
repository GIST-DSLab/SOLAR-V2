"""
ARC Task: 08ed6ac7 (RE-ARC) — LLM-generated grid_maker
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
    # bgc is the only randomly sampled colour that matters for learnability:
    # the rule (rank bars by height -> colours 1..4) is independent of the bar
    # colours, but the background must be constant across the episode.
    # The generator forbids bgc in {1,2,3,4}, keep that invariant.
    bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 4)])
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    barrange = (4, w)
    locopts = interval(0, w, 1)
    nbars = unifint(diff_lb, diff_ub, barrange)
    barlocs = sample(locopts, nbars)
    barhopts = interval(0, h, 1)
    barhs = sample(barhopts, 4)
    barcols = [choice(remcols) for j in range(nbars)]
    barhsfx = [choice(barhs) for j in range(nbars - 4)] + list(barhs)
    shuffle(barhsfx)
    ordered = sorted(barhs)
    colord = interval(1, 5, 1)
    for col, (loci, locj) in zip(barcols, list(zip(barhsfx, barlocs))):
        bar = connect((loci, locj), (h - 1, locj))
        gi = fill(gi, col, bar)
        go = fill(go, colord[ordered.index(loci)], bar)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule: every bar is a 1-wide vertical segment that runs down to the last row.
    There are exactly 4 distinct bar lengths; the longest class becomes 1, then
    2, 3, and the shortest class becomes 4.

    Route: bars of equal length are IDENTICAL objects (same rows, same width,
    and in the output the same colour).  So per length-class we recolour ONE
    representative bar, CopyO it, and Paste that recoloured bar onto every other
    bar of that class -- the replication the rule is made of, performed rather
    than re-painted cell by cell.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # ---- find the bars in I -------------------------------------------------
    def bars_for(bg):
        """Bars assuming background colour `bg`; None if inconsistent."""
        found = []
        for c in range(w):
            col = I[:, c]
            if int(col[h - 1]) == bg:
                if np.any(col != bg):
                    return None          # a bar not touching the bottom row
                continue
            x = int(col[h - 1])
            r = h - 1
            while r - 1 >= 0 and int(col[r - 1]) == x:
                r -= 1
            if np.any(col[:r] != bg):
                return None              # something above the bar
            found.append((c, r, h - r))  # (column, top row, length)
        return found

    bars = None
    for v, _ in Counter(I.flatten().tolist()).most_common():
        if v in (1, 2, 3, 4):            # generator never uses these as bgc
            continue
        cand = bars_for(v)
        if cand is not None and len(cand) >= 4 and len({t[2] for t in cand}) == 4:
            bars = cand
            break

    if bars is None:                     # defensive fallback
        bars = []
        for c in range(w):
            rows = [r for r in range(h) if int(O[r, c]) in (1, 2, 3, 4)]
            if rows:
                bars.append((c, min(rows), len(rows)))

    # ---- rank the length classes: longest -> 1 ... shortest -> 4 ------------
    lengths = sorted({L for (_, _, L) in bars}, reverse=True)
    rank = {L: i + 1 for i, L in enumerate(lengths)}

    # ---- recolour one bar per class, then replicate it onto its twins -------
    for L in lengths:
        target = rank[L]
        group = sorted([b for b in bars if b[2] == L], key=lambda t: t[0])
        c0, r0, _ = group[0]
        src_cells = [(r, c0) for r in range(r0, r0 + L)]

        # paint the representative bar to its rank colour (skip if already so)
        if int(I[r0, c0]) != target:
            ops.append(target)
            sels.append(sel_of(src_cells))

        twins = [b for b in group[1:] if int(I[b[1], b[0]]) != target]
        if twins:
            ops.append(29)               # CopyO: grab the just-recoloured bar
            sels.append(sel_of(src_cells))
            for (c2, r2, _) in twins:    # same length => same top row
                ops.append(30)           # Paste at the twin's top cell
                sels.append(sel_of([(r2, c2)]))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])    # full grid = exactly this rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 08ed6ac7"
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
                                f"for task 08ed6ac7"
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
                    f"Failed to build a complete episode for task 08ed6ac7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"08ed6ac7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
