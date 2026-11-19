"""
ARC Task: 941d9a10 (RE-ARC) — LLM-generated grid_maker
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
    # Colours 1, 2 and 3 are reserved for the output marks. The generator draws
    # the background and line colours from the other seven.
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(diff_lb, diff_ub)
    v = lo + int(round((hi - lo) * d))
    return max(lo, min(hi, v))


def _components(G, color):
    hh, ww = G.shape
    seen = np.zeros((hh, ww), dtype=bool)
    comps = []
    for r in range(hh):
        for c in range(ww):
            if G[r, c] == color and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hh and 0 <= nc < ww and G[nr, nc] == color and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(cells)
    return comps


def _roles(comps):
    info = []
    for cells in comps:
        info.append({
            'cells': cells,
            'left': min(p[1] for p in cells),
            'up': min(p[0] for p in cells),
        })
    blue = min(info, key=lambda o: o['left'] + o['up'])
    green = max(info, key=lambda o: o['left'] + o['up'])

    def red_ok(o):
        lt = sum(1 for x in info if x['left'] < o['left'])
        gt = sum(1 for x in info if x['left'] > o['left'])
        ut = sum(1 for x in info if x['up'] < o['up'])
        dt = sum(1 for x in info if x['up'] > o['up'])
        return lt == gt and ut == dt
    red = next(o for o in info if red_ok(o))
    return blue, red, green


def _axis_lines(n_total, diff_lb, diff_ub):
    # Return a boolean list of length n_total marking which indices are lines.
    opts = list(range(2, (n_total - 1) // 2 + 1, 2))
    idx = _unifint(diff_lb, diff_ub, 0, len(opts) - 1)
    nl = opts[idx]
    base = [False] * (2 * nl + 1)
    for j in range(1, 2 * nl + 1, 2):
        base[j] = True
    for _ in range(n_total - (2 * nl + 1)):
        loc = random.randint(0, len(base) - 1)
        base = base[:loc] + [False] + base[loc:]
    return base


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    if bgc is None or fgc is None:
        cols = [c for c in range(10) if c not in (1, 2, 3)]
        bgc, fgc = random.sample(cols, 2)
    h = _unifint(diff_lb, diff_ub, 5, max(5, max_h))
    w = _unifint(diff_lb, diff_ub, 5, max(5, max_w))
    rows = _axis_lines(h, diff_lb, diff_ub)
    colsl = _axis_lines(w, diff_lb, diff_ub)
    gi = np.full((h, w), bgc, dtype=int)
    for r in range(h):
        if rows[r]:
            gi[r, :] = fgc
    for c in range(w):
        if colsl[c]:
            gi[:, c] = fgc
    comps = _components(gi, bgc)
    blue, red, green = _roles(comps)
    go = gi.copy()
    for o, col in ((blue, 1), (green, 3), (red, 2)):
        for (r, c) in o['cells']:
            go[r, c] = col
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # Every grid corner lies inside a background cell, never on a grid line.
    corner_cells = [I[0, 0], I[0, wi - 1], I[hi - 1, 0], I[hi - 1, wi - 1]]
    bgc = Counter(int(x) for x in corner_cells).most_common(1)[0][0]

    # The lines split the background into rooms (connected regions read from I).
    comps = _components(I, bgc)
    blue, red, green = _roles(comps)

    # Read each room as one floodable region and colour it from a single seed.
    # Work from the top-left corner room, through the middle room, to the
    # bottom-right corner room.
    ops, sels = [], []
    # Top-left corner room -> blue. (0,0) is always inside it.
    ops.append(10 + 1); sels.append(sel_of([(0, 0)]))
    # Middle room (same number of rooms on each side in both directions) -> red.
    # Seed it at its top-left cell.
    seed = (red['up'], red['left'])
    ops.append(10 + 2); sels.append(sel_of([seed]))
    # Bottom-right corner room -> green, seeded at the grid's last cell.
    ops.append(10 + 3); sels.append(sel_of([(hi - 1, wi - 1)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 941d9a10"
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
                                f"for task 941d9a10"
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
                    f"Failed to build a complete episode for task 941d9a10 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"941d9a10-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
