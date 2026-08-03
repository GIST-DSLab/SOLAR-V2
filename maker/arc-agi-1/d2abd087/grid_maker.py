"""
ARC Task: d2abd087 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        cells = list(cells)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]


def sample_colors(num_examples=None) -> dict:
    # bgc: generator picks any color except 1 and 2 (those are the output labels)
    bgc = random.choice([c for c in range(10) if c not in (1, 2)])
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        opts = asindices(canvas(-1, (5, 5)))
        sp = choice(totuple(opts))
        opts = remove(sp, opts)
        lb = unifint(diff_lb, diff_ub, (1, 5))
        lopts = interval(lb, 6, 1)
        ubi = unifint(diff_lb, diff_ub, (1, 5))
        ub = 12 - ubi
        uopts = interval(7, ub + 1, 1)
        if choice((True, False)):
            numcells = 6
        else:
            numcells = choice(lopts + uopts)
        obj = {sp}
        for k in range(numcells - 1):
            cands = totuple((opts - obj) & mapply(dneighbors, obj))
            if len(cands) == 0:
                break
            obj.add(choice(cands))
        obj = normalize(obj)
        loc = choice(totuple(inds))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            gi = fill(gi, choice(remcols), plcd)
            go = fill(go, 1 + (len(obj) == 6), plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        tr += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Background: the colour the canvas is painted with before objects are placed.
    # It is never 1 or 2 (those are reserved as the output labels), and it dominates.
    counts = Counter(I.flatten().tolist())
    bgc = None
    for col, _ in counts.most_common():
        if col not in (1, 2):
            bgc = col
            break
    if bgc is None:
        bgc = counts.most_common(1)[0][0]

    # Segment I into univalued, 4-connected, non-background objects.
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] == col:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((col, sorted(cells)))

    ops, sels = [], []

    # Rule measured from I: an object of exactly 6 cells becomes 2, every other size becomes 1.
    for col, cells in comps:
        tgt = 2 if len(cells) == 6 else 1
        if col == tgt:
            continue  # object already holds its label colour; a fill would be a no-op
        sr, sc = cells[0]
        ops.append(10 + tgt)          # FloodFill<tgt> — objects are never 4-adjacent
        sels.append([sr, sc, 0, 0])   # single seed cell (mask has exactly 1 True)

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
                        f"num_examples+1 ({num_examples + 1}) for task d2abd087"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d2abd087"
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
                                f"for task d2abd087"
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
                    f"Failed to build a complete episode for task d2abd087 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d2abd087-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
