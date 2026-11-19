"""
ARC Task: bdad9b1f (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 4]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))
    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    numh = unifint(diff_lb, diff_ub, (1, h // 2 - 1))
    numw = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
    hlocs = random.sample(range(2, h - 1), numh)
    wlocs = random.sample(range(2, w - 1), numw)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, numcols)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    fc = -1
    for ii in sorted(hlocs):
        col = random.choice([c for c in ccols if c != fc]); fc = col
        objw = random.randint(2, max(2, min(ii, min(wlocs))))  # never reach a vertical line
        gi[ii, 0:objw] = col
        go[ii, :] = col
    fc = -1
    for jj in sorted(wlocs):
        col = random.choice([c for c in ccols if c != fc]); fc = col
        objh = random.randint(2, max(2, min(jj, min(hlocs))))  # never reach a horizontal line
        gi[0:objh, jj] = col
        go[:, jj] = col
    for ii in hlocs:
        for jj in wlocs:
            go[ii, jj] = 4
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, -k); go = np.rot90(go, -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the generator paints the canvas first; demonstrations share it
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # find line segments (single-colour 4-connected components, 1 cell thick)
    seen = np.zeros_like(I, dtype=bool)
    hlines, vlines = [], []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c]); st = [(r, c)]; cells = []
            seen[r, c] = True
            while st:
                y, x = st.pop(); cells.append((y, x))
                for ny, nx in ((y+1, x), (y-1, x), (y, x+1), (y, x-1)):
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True; st.append((ny, nx))
            rs = {y for y, _ in cells}; cs = {x for _, x in cells}
            if len(rs) == 1 and len(cs) >= 2:
                hlines.append((cells[0][0], col))
            elif len(cs) == 1 and len(rs) >= 2:
                vlines.append((cells[0][1], col, cells))

    ops, sels = [], []
    g = I.copy()
    hrows = {r for r, _ in hlines}

    # Reading: the edge row the vertical lines grow from is a template;
    # every ordinary row becomes a copy of it. Horizontal-line rows are
    # painted whole afterwards, and their crossings turn yellow.
    if vlines:
        top = any(y == 0 for _, _, cells in vlines for y, _ in cells)
        t = 0 if top else hi - 1
        tmpl = I[t].copy()
        ops.append(28); sels.append(sel_of([(t, c) for c in range(wi)]))  # whole template row
        for r in range(hi):
            if r == t or r in hrows:
                continue
            nz = [c for c in range(wi) if tmpl[c] != 0]
            if any(g[r, c] != tmpl[c] for c in nz):
                ops.append(30); sels.append(sel_of([(r, 0)]))
                for c in nz:
                    g[r, c] = tmpl[c]
        # Copy/Paste cannot carry colour 0: extend 0-coloured vertical lines by colouring their column
        for j, col, _ in vlines:
            if col == 0:
                cells = [(r, j) for r in range(hi) if r not in hrows]
                if any(g[r, c] != 0 for r, c in cells):
                    ops.append(0); sels.append(sel_of(cells))
                    for r, c in cells:
                        g[r, c] = 0

    # horizontal lines extend across the full row
    for r, col in hlines:
        cells = [(r, c) for c in range(wi)]
        if all(g[y, x] == col for y, x in cells):
            continue
        ops.append(col); sels.append(sel_of(cells))
        for _, c in cells:
            g[r, c] = col

    # crossings of horizontal and vertical lines become 4
    cross = [(r, j) for r, _ in hlines for j, _, _ in vlines]
    if cross:
        ops.append(4); sels.append(sel_of(cross))
        for r, c in cross:
            g[r, c] = 4

    ops.append(34); sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
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
                                f"for task bdad9b1f"
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
                    f"Failed to build a complete episode for task bdad9b1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bdad9b1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
