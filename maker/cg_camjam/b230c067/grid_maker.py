"""
ARC Task: b230c067 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque, Counter

import numpy as np

try:
    from dsl import *
except Exception:
    pass
try:
    from utils import unifint
except Exception:
    pass

from maker.sel_helpers import sel_of

# Reading of the task:
# - Every figure in the input has the same colour.
# - Each figure starts as colour 1.
# - The figure whose size (cell count) is shared by no other figure is then
#   raised to colour 2.
# The odd figure (shpa) can be either smaller or larger than the repeated
# pair (shpb), depending on the generator's swap. The episode plan includes
# both cases, so "smallest figure" is never a usable shortcut.

VARIANTS = [{"swap": True}, {"swap": False}]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc, fgc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def _components8(grid, bgc):
    g = np.asarray(grid, dtype=int)
    hh, ww = g.shape
    seen = np.zeros((hh, ww), dtype=bool)
    comps = []
    for r in range(hh):
        for c in range(ww):
            if g[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hh and 0 <= nx < ww and not seen[ny, nx] and g[ny, nx] != bgc:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            comps.append(cells)
    return comps


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, swap=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    if swap is None:
        swap = random.choice([True, False])
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (10, max_h))
        w = unifint(diff_lb, diff_ub, (10, max_w))
        oh = unifint(diff_lb, diff_ub, (2, h // 3 - 1))
        ow = unifint(diff_lb, diff_ub, (2, w // 3 - 1))
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bounds))
        shp = {sp}
        for k in range(numca):
            ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
            shp.add(ij)
        shpa = normalize(shp)
        shpb = set(normalize(shp))
        mxnch = oh * ow - len(shpa)
        if mxnch < 1:
            continue
        nchinv = unifint(diff_lb, diff_ub, (1, mxnch))
        nch = mxnch - nchinv
        nch = min(max(1, nch), mxnch)
        for k in range(nch):
            ij = choice(totuple((bounds - shpb) & mapply(neighbors, shpb)))
            shpb.add(ij)
        shpb = frozenset(shpb)
        if swap:
            shpa, shpb = shpb, shpa
        c = canvas(bgc, (h, w))
        inds = asindices(c)
        acands = sfilter(inds, lambda ij: ij[0] <= h - height(shpa) and ij[1] <= w - width(shpa))
        aloc = choice(totuple(acands))
        aplcd = shift(shpa, aloc)
        gi = fill(c, fgc, aplcd)
        go = fill(c, 2, aplcd)
        maxtrials = 10
        tr = 0
        succ = 0
        inds = (inds - aplcd) - mapply(neighbors, aplcd)
        inds = sfilter(inds, lambda ij: ij[0] <= h - height(shpb) and ij[1] <= w - width(shpb))
        while succ < 2 and tr <= maxtrials:
            if len(inds) == 0:
                break
            loc = choice(totuple(inds))
            plcbd = shift(shpb, loc)
            if plcbd.issubset(inds):
                gi = fill(gi, fgc, plcbd)
                go = fill(go, 1, plcbd)
                succ += 1
                inds = (inds - plcbd) - mapply(neighbors, plcbd)
            tr += 1
        if succ != 2:
            continue
        # reject diagonal contact: the three figures must be three separate 8-connected objects
        if len(_components8(gi, bgc)) != 3:
            continue
        break
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = the canvas colour (figures are small: at most ~1/9 of the grid each)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    comps = _components8(I, bgc)

    # group figures by size; the odd one out is the size nobody else shares
    by_size = {}
    for cells in comps:
        by_size.setdefault(len(cells), []).append(cells)
    odd_size = min(by_size, key=lambda s: len(by_size[s]))
    odd = by_size[odd_size]

    ops, sels = [], []
    # base layer: every figure cell (all cells of the single figure colour) -> 1
    fg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    ops.append(1); sels.append(sel_of(fg_cells))
    # overlay: the figure whose size is unique is raised to 2
    for cells in odd:
        ops.append(2); sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task b230c067"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b230c067"
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
                                f"for task b230c067"
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
                    f"Failed to build a complete episode for task b230c067 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b230c067-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
