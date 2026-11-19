"""
ARC Task: b1948b0a (RE-ARC) — LLM-generated grid_maker
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
    # Every color here is hardcoded in the generator: the marker/background is 6,
    # the replacement is 2, and each remaining cell takes an arbitrary color from
    # 0..9 minus 6. The rule depends only on the presence of 6, so nothing to fix.
    return {}


def generate(diff_lb: float, diff_ub: float, max_h: int = 30, max_w: int = 30) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    npd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    num = min(max(1, npd), h * w)          # at least one 6-cell so I != O
    gi = canvas(6, (h, w))
    inds = totuple(asindices(gi))
    pp = sample(inds, num)
    npp = difference(inds, pp)
    for ij in npp:
        gi = fill(gi, choice(cols), {ij})
    go = fill(gi, 2, pp)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Repaint every 6 as a 2.

    The rule is one substitution, so nothing in it forces an order: a solver may
    sweep the grid in bands, deal with the connected clumps of 6 before the
    strays, or take the lot in one stroke. Which of those happens is read off the
    input alone (its size, its 6-count, how those 6s clump, the color in its
    top-left corner) — handed a different answer grid, this returns the same ops.
    Every op paints a non-empty set of 6-cells that no other op touches, so no
    op here can be dropped.
    """
    import numpy as _np
    from maker.sel_helpers import sel_of

    I = _np.asarray(I, dtype=int)
    O = _np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    MARK, NEW = 6, 2
    ops, sels = [], []

    cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == MARK]
    if not cells:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    n6 = len(cells)
    cellset = set(cells)

    # the 6s as the eye groups them: 4-connected clumps, and lone pixels
    comps, seen = [], set()
    for start in cells:
        if start in seen:
            continue
        seen.add(start)
        stack, comp = [start], []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nb in cellset and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(sorted(comp))
    blobs = [k for k in comps if len(k) > 1]
    dots = [k[0] for k in comps if len(k) == 1]

    routes = ["rows", "cols", "whole"]
    if blobs and len(blobs) <= 24:
        routes.append("blobs")
    route = routes[(hi * 7 + wi * 13 + n6 * 3 + len(comps) + int(I[0, 0])) % len(routes)]

    if route in ("rows", "cols"):
        # sweep the grid in bands, one stroke per band
        span = hi if route == "rows" else wi
        nb = max(1, min(2 + (n6 % 3), span))
        cut = [(k * span) // nb for k in range(nb + 1)]
        bands = []
        for k in range(nb):
            lo, up = cut[k], cut[k + 1]
            g = [(r, c) for (r, c) in cells
                 if lo <= (r if route == "rows" else c) < up]
            if g:
                bands.append(g)
        if (hi + wi + n6) % 2:
            bands.reverse()              # sweep from the far edge back instead
        for g in bands:
            ops.append(NEW)              # Color2 over this band's 6s
            sels.append(sel_of(g))

    elif route == "blobs":
        # take the clumps one at a time, then the strays in a single stroke
        if n6 % 2:
            order = sorted(blobs, key=lambda b: (-len(b), b[0]))   # biggest first
        else:
            order = sorted(blobs, key=lambda b: b[0])              # reading order
        for b in order:
            if len(b) <= 120:
                ops.append(10 + NEW)     # FloodFill2 from one cell of the clump
                sels.append(sel_of([b[0]]))
            else:
                ops.append(NEW)          # clump too deep for ARCLE's fill: one stroke
                sels.append(sel_of(b))
        if dots:
            ops.append(NEW)              # Color2 over the leftover lone 6s
            sels.append(sel_of(dots))

    else:
        ops.append(NEW)                  # one stroke over every 6
        sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task b1948b0a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b1948b0a"
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
                                f"for task b1948b0a"
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
                    f"Failed to build a complete episode for task b1948b0a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b1948b0a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
