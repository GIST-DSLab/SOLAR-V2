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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # Only the noise colour is a single sampled role. The region colours are drawn
    # separately for each instance by generate(). The rule does not depend on which
    # colours are used.
    noisec = random.choice(range(10))
    return {"noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    def comps_of(g):
        h, w = g.shape
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c]:
                    continue
                col = g[r, c]
                seen[r, c] = True
                st = [(r, c)]
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
                out.append((int(col), cells))
        return out

    if noisec is None:
        noisec = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (6, max(6, min(30, max_w))))
    remcols = [c for c in range(10) if c != noisec]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = np.full((h, w), starterc, dtype=int)
    for k in range(ncols - 1):
        objs = []
        for col, cells in comps_of(gi):
            rs = [r for r, _ in cells]; cs = [c for _, c in cells]
            oh = max(rs) - min(rs) + 1; ow = max(cs) - min(cs) + 1
            if oh > 5 or ow > 5:
                objs.append((cells, min(rs), max(rs), min(cs), max(cs), oh, ow))
        if not objs:
            break
        cells, u, lo, l, ri, oh, ow = random.choice(objs)
        if oh > 5 and ow > 5:
            ax = random.choice((0, 1))
        elif oh > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = random.randint(u + 3, lo - 2)
            newobj = [(r, c) for r, c in cells if r >= loci]
        else:
            locj = random.randint(l + 3, ri - 2)
            newobj = [(r, c) for r, c in cells if c >= locj]
        for r, c in newobj:
            gi[r, c] = ccols[k]
    objs = sorted(comps_of(gi), key=lambda o: len(o[1]))
    info = []
    for col, cells in objs:
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        info.append((col, min(rs), max(rs), min(cs), max(cs)))
    allowances = [max(1, ((b - a + 1 - 2) * (d - c + 1 - 2)) // 2) for _, a, b, c, d in info]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [random.randint(0, min(meann, a)) for a in allowances]
    while max(chosens) == 0:
        chosens = [random.randint(0, min(meann, a)) for a in allowances]
    mx = max(chosens)
    fixinds = [i for i, cnt in enumerate(chosens) if cnt == mx]
    gogocol = info[fixinds[0]][0]
    for i in fixinds[1:]:
        chosens[i] -= 1
    for (col, a, b, c, d), cnt in zip(info, chosens):
        inner = [(r, cc) for r in range(a + 1, b) for cc in range(c + 1, d)]
        if cnt > len(inner):
            cnt = len(inner)
        for r, cc in random.sample(inner, cnt):
            gi[r, cc] = noisec
    go = np.full((1, 1), gogocol, dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading used here (from the code-golf entry): treat each COLOUR as one unit, not
    each connected component. For colour X, count the cells inside X's row/column
    span (its bounding rectangle) that are not X. These are the 'holes' punched
    into X's rectangle. The colour with the most holes wins.
    The noise colour is not a region. It is identified from I alone, the same way
    the verifier does, and left out.
    Ops: (1) CropGrid to the winning colour's rectangle. This is a named object
    and shows which region was chosen. (2) CropGrid to that rectangle's corner
    cell. Noise only lands in a region's interior, so the corner always holds the
    region's colour. (3) Submit.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # connected components (used only to identify the noise colour, as the verifier does)
    seen = np.zeros((hi, wi), dtype=bool)
    comp_colors = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c]:
                continue
            col = int(I[r, c]); seen[r, c] = True
            st = [(r, c)]
            while st:
                y, x = st.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        st.append((ny, nx))
            comp_colors.append(col)
    if len(set(comp_colors)) == len(comp_colors):
        cnt = Counter(I.flatten().tolist())
        noise = min(cnt.items(), key=lambda kv: kv[1])[0]
    else:
        noise = Counter(comp_colors).most_common(1)[0][0]

    # per colour: holes = non-X cells inside X's rows x columns span
    best = None
    for X in sorted(set(I.flatten().tolist())):
        if X == noise:
            continue
        rows = np.where((I == X).any(axis=1))[0]
        cols = np.where((I == X).any(axis=0))[0]
        r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
        holes = int(np.count_nonzero(I[r0:r1 + 1, c0:c1 + 1] != X))
        if best is None or holes > best[0]:
            best = (holes, int(r0), int(c0), int(r1 - r0), int(c1 - c0))

    _, r0, c0, dh, dw = best
    ops, sels = [], []
    # 1. crop to the winning colour's full rectangle (exact bbox of that region)
    ops.append(33); sels.append([r0, c0, dh, dw])
    # 2. crop to the rectangle's corner cell (always the region's colour)
    ops.append(33); sels.append([0, 0, 0, 0])
    ops.append(34); sels.append([0, 0, 0, 0])
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
