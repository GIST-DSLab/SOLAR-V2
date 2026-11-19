"""
ARC Task: f8c80d96 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# How this differs from the maker:
# - Both readings agree on the rule: the input holds the inner rings of a family of
#   nested boxes. The output draws every ring that still touches the grid in the line
#   colour and turns everything else into 5.
# - The code-golf entry treats the output as one field: a cell's colour depends only on
#   its distance from the existing lines, counted in steps of the ring spacing. So the
#   finished picture is "complete the line pattern, and whatever is not a line is 5".
# - The maker turned the background to 5 first and then stamped the missing rings on
#   top. This route follows the golf order instead:
#     (a) stamp each missing ring in the line colour, from the innermost missing ring
#         outward, one op per ring;
#     (b) recolour every cell still holding the background colour to 5. That selection
#         is "all cells of one colour" in the current grid.
# - Every ring is computed from the input alone (centre box, row/col spacing), and the
#   line colour is the one the demonstrations keep unchanged. The answer grid is never read.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 5]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _box(r0, c0, r1, c1):
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c)); cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0)); cells.add((r, c1))
    return cells


def _rings(h, w, loci, locj, oh, ow, hoffs, woffs):
    ulci, ulcj = loci - 1, locj - 1
    lrci, lrcj = loci + oh, locj + ow
    out = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        b = _box(ulci - hoffs * k, ulcj - woffs * k, lrci + hoffs * k, lrcj + woffs * k)
        b = {(r, c) for r, c in b if 0 <= r < h and 0 <= c < w}
        if b:
            out.append(sorted(b))
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kw) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        b = max(a, b)
        lo = a + int(round((b - a) * lb)); hi = a + int(round((b - a) * ub))
        return random.randint(min(lo, hi), max(lo, hi))
    if bgc is None or linc is None:
        bgc, linc = random.sample([c for c in range(10) if c != 5], 2)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
    ow = random.randint(1, 3 if h > 10 else 2)
    oh = random.randint(1, 3 if w > 10 else 2)
    loci = random.randint(-oh + 1, h - 1)
    locj = random.randint(-ow + 1, w - 1)
    hoffs = random.randint(2, 4 if h > 12 else 3)
    woffs = random.randint(2, 4 if w > 12 else 3)
    lns = _rings(h, w, loci, locj, oh, ow, hoffs, woffs)
    nlns = len(lns)
    nmissing = unifint(diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    gi = [[bgc] * w for _ in range(h)]
    go = [[5] * w for _ in range(h)]
    for k in range(npresent):
        for r, c in lns[k]:
            gi[r][c] = linc
    for ln in lns:
        for r, c in ln:
            go[r][c] = linc
    return {"input": gi, "output": go}


def _fit(I, linc):
    """Recover the box family (centre box, row/col spacing) whose first few
    rings are exactly the line-coloured cells of I."""
    h, w = I.shape
    lines = {(r, c) for r in range(h) for c in range(w) if I[r, c] == linc}
    if not lines:
        return None
    for oh in range(1, 4):
        for ow in range(1, 4):
            for loci in range(-oh + 1, h):
                for locj in range(-ow + 1, w):
                    r0 = _box(loci - 1, locj - 1, loci + oh, locj + ow)
                    r0 = [(r, c) for r, c in r0 if 0 <= r < h and 0 <= c < w]
                    if not r0 or any(I[r, c] != linc for r, c in r0):
                        continue
                    for hoffs in range(2, 5):
                        for woffs in range(2, 5):
                            rings = _rings(h, w, loci, locj, oh, ow, hoffs, woffs)
                            acc = set()
                            for k, rg in enumerate(rings):
                                acc |= set(rg)
                                if not acc <= lines:
                                    break
                                if acc == lines:
                                    if k + 1 >= 2:
                                        return rings, k + 1
                                    break
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    present = sorted(set(I.flatten().tolist()))

    # Line colour = the colour the demonstrations keep unchanged; background = the other.
    cand = []
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            for c in set(ei.flatten().tolist()):
                if np.all(eo[ei == c] == c) and c not in cand:
                    cand.append(int(c))
    cand = [c for c in cand if c in present] + [c for c in present if c not in cand]

    fit = linc = bgc = None
    for c in cand:
        others = [x for x in present if x != c]
        if len(others) != 1:
            continue
        fit = _fit(I, c)
        if fit is not None:
            linc, bgc = c, others[0]
            break
    if fit is None:
        raise ValueError("no box family fits the input")
    rings, npresent = fit

    # 1) continue the nested box family outward, one stamp per missing ring
    for k in range(npresent, len(rings)):
        ops.append(int(linc)); sels.append(sel_of(rings[k]))

    # 2) every cell still holding the background colour becomes 5
    grid = I.copy()
    for k in range(npresent, len(rings)):
        for r, c in rings[k]:
            grid[r, c] = linc
    bg_cells = [(r, c) for r in range(h) for c in range(w) if grid[r, c] == bgc]
    if bg_cells:
        ops.append(5); sels.append(sel_of(bg_cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
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
                                f"for task f8c80d96"
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
                    f"Failed to build a complete episode for task f8c80d96 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8c80d96-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
