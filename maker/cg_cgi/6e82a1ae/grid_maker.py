"""
ARC Task: 6e82a1ae (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of

# Object size -> output colour (fixed by the task)
SIZE_TO_COLOR = {2: 3, 3: 2, 4: 1}


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def _dnb(cells):
    out = set()
    for r, c in cells:
        out.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
    return out


def _normalize(shp):
    mr = min(r for r, _ in shp)
    mc = min(c for _, c in shp)
    return frozenset((r - mr, c - mc) for r, c in shp)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])
    b = [frozenset({(0, 0), (0, 1)}), frozenset({(0, 0), (1, 0)})]
    c = [
        frozenset({(0, 0), (1, 0), (1, 1)}),
        frozenset({(0, 0), (1, 0), (0, 1)}),
        frozenset({(1, 1), (1, 0), (0, 1)}),
        frozenset({(1, 1), (0, 0), (0, 1)}),
        frozenset({(0, 0), (1, 0), (2, 0)}),
        frozenset({(0, 0), (0, 1), (0, 2)}),
    ]
    d = set()
    for _ in range(100):
        shp = {(0, 0)}
        for _ in range(3):
            shp.add(random.choice(sorted(_dnb(shp) - shp)))
        d.add(_normalize(shp))
    d = sorted(d, key=lambda s: sorted(s))
    prs = [(b, 3), (c, 2), (d, 1)]
    h = _unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = _unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    reminds = {(r, cc) for r in range(h) for cc in range(w)}
    nobjs = _unifint(diff_lb, diff_ub, (1, max(1, ((h * w) // 2) // 3)))
    for _ in range(nobjs):
        objs, col = random.choice(prs)
        obj = random.choice(objs)
        for _ in range(10):
            if not reminds:
                break
            lr, lc = random.choice(sorted(reminds))
            ol = {(r + lr, cc + lc) for r, cc in obj}
            if ol <= reminds:
                for r, cc in ol:
                    gi[r][cc] = fgc
                    go[r][cc] = col
                reminds = (reminds - ol) - _dnb(ol)
                break
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(color):
        seen = np.zeros((hi, wi), dtype=bool)
        comps = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != color or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                and I[nr, nc] == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                comps.append(cells)
        return comps

    present = sorted(set(I.flatten().tolist()))
    comps_by_color = {col: components(col) for col in present}
    # Background: the colour that has a large connected region (objects have <= 4 cells)
    bgc, best = None, -1
    for col in present:
        mx = max((len(cm) for cm in comps_by_color[col]), default=0)
        if mx > best:
            best, bgc = mx, col
    if best <= 4:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Each object is one connected unit; its cell count decides its colour.
    # Visit objects one at a time in reading order and flood-fill each from
    # one seed cell, so the connected component is what gets recoloured.
    objs = []
    for col in present:
        if col != bgc:
            objs.extend(comps_by_color[col])
    objs.sort(key=lambda ob: min(ob))

    ops, sels = [], []
    for cells in objs:
        new_col = SIZE_TO_COLOR.get(len(cells))
        if new_col is None:
            continue
        seed = min(cells)
        ops.append(10 + new_col)          # FloodFill the whole object
        sels.append([seed[0], seed[1], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6e82a1ae"
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
                                f"for task 6e82a1ae"
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
                    f"Failed to build a complete episode for task 6e82a1ae "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6e82a1ae-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
