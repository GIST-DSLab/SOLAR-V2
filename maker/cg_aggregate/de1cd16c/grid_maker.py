"""
ARC Task: de1cd16c (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule is structural: noise is the scattered color, and the winner is the region
    # holding the most noise. So the colors only fix the episode's palette.
    cols = list(range(10))
    noisec = random.choice(cols)
    remcols = [c for c in cols if c != noisec]
    random.shuffle(remcols)
    return {"noisec": noisec, "region_cols": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, region_cols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    def components(g):
        H, W = g.shape
        seen = np.zeros((H, W), dtype=bool)
        out = []
        for r in range(H):
            for c in range(W):
                if seen[r, c]:
                    continue
                col = int(g[r, c])
                seen[r, c] = True
                st = [(r, c)]
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((col, cells))
        return out

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    cols = list(range(10))
    if noisec is None:
        noisec = random.choice(cols)
    if region_cols is None:
        region_cols = [c for c in cols if c != noisec]
        random.shuffle(region_cols)
    region_cols = [c for c in region_cols if c != noisec]

    h = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_w))))
    ncols = unifint(diff_lb, diff_ub, (2, min(9, len(region_cols))))
    ccols = region_cols[:ncols]
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = np.full((h, w), starterc, dtype=int)
    for k in range(ncols - 1):
        objs = []
        for col, cells in components(gi):
            u, d, l, r = bbox(cells)
            if (d - u + 1) > 5 or (r - l + 1) > 5:
                objs.append((col, cells))
        if len(objs) == 0:
            break
        col, cells = random.choice(objs)
        u, d, l, r = bbox(cells)
        oh, ow = d - u + 1, r - l + 1
        if oh > 5 and ow > 5:
            ax = random.choice((0, 1))
        elif oh > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = random.randint(u + 3, d - 2)
            newobj = [(y, x) for y, x in cells if y >= loci]
        else:
            locj = random.randint(l + 3, r - 2)
            newobj = [(y, x) for y, x in cells if x >= locj]
        for y, x in newobj:
            gi[y, x] = ccols[k]

    objs = sorted(components(gi), key=lambda o: len(o[1]))
    allowances = []
    for col, cells in objs:
        u, d, l, r = bbox(cells)
        allowances.append(max(1, ((d - u + 1 - 2) * (r - l + 1 - 2)) // 2))
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [random.randint(0, min(meann, a)) for a in allowances]
    while max(chosens) == 0:
        chosens = [random.randint(0, min(meann, a)) for a in allowances]
    mx = max(chosens)
    fixinds = [i for i, cnt in enumerate(chosens) if cnt == mx]
    gogocol = objs[fixinds[0]][0]
    for i in fixinds[1:]:
        chosens[i] -= 1
    for (col, cells), cnt in zip(objs, chosens):
        u, d, l, r = bbox(cells)
        inner = [(y, x) for y in range(u + 1, d) for x in range(l + 1, r)]
        if cnt > len(inner):
            cnt = len(inner)
        for y, x in random.sample(inner, cnt):
            gi[y, x] = noisec
    go = np.full((1, 1), gogocol, dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    The rule, read from I: the grid is tiled with solid rectangular regions, and one more
    color is scattered over them as noise. The winner is the region whose bounding box holds
    the most noise pixels, and the answer is that region's color.
    The code-golf reading (count row/column color co-occurrences, then pick by rank) still
    comes down to "pick one region's color". So both readings give the same trajectory:
    crop the canvas to one cell of the winning region (that cell is chosen from I), then Submit.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            st = [(r, c)]
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            comps.append((col, cells))

    comp_colors = [col for col, _ in comps]
    if len(set(comp_colors)) == len(comp_colors):
        cnt = Counter(I.flatten().tolist())
        noise = min(cnt.items(), key=lambda kv: kv[1])[0]
    else:
        noise = Counter(comp_colors).most_common(1)[0][0]

    best_cells, best_count = None, -1
    for col, cells in comps:
        if col == noise:
            continue
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        sub = I[min(rs):max(rs) + 1, min(cs):max(cs) + 1]
        n = int(np.count_nonzero(sub == noise))
        if n > best_count:
            best_count, best_cells = n, cells

    if best_cells is None:
        best_cells = comps[0][1]
    r, c = best_cells[0]  # a cell of the winning region itself, found from I

    ops, sels = [], []
    ops.append(33); sels.append(sel_of([(int(r), int(c))]))   # crop the canvas to that single cell
    ops.append(34); sels.append([0, 0, 0, 0])                 # whole 1x1 grid
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
                        f"num_examples+1 ({num_examples + 1}) for task de1cd16c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task de1cd16c"
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
                                f"for task de1cd16c"
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
                    f"Failed to build a complete episode for task de1cd16c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"de1cd16c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
