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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    noisec = random.choice(list(range(10)))
    return {"noisec": noisec}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i = int(round(lo))
    hi_i = int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, lo_i), min(b, hi_i))


def _objects(g):
    g = np.asarray(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objs = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = g[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            objs.append((int(col), cells))
    return objs


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, **kwargs) -> dict:
    cols = list(range(10))
    if noisec is None:
        noisec = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, (6, max_h))
    w = _unifint(diff_lb, diff_ub, (6, max_w))
    remcols = [c for c in cols if c != noisec]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = np.full((h, w), starterc, dtype=int)
    for k in range(ncols - 1):
        objs = _objects(gi)
        def hw(cells):
            rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
            return min(rs), max(rs), min(cs), max(cs)
        cand = []
        for col, cells in objs:
            r0, r1, c0, c1 = hw(cells)
            if (r1 - r0 + 1) > 5 or (c1 - c0 + 1) > 5:
                cand.append((cells, r0, r1, c0, c1))
        if not cand:
            break
        cells, r0, r1, c0, c1 = random.choice(cand)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        if oh > 5 and ow > 5:
            ax = random.choice((0, 1))
        elif oh > 5:
            ax = 0
        else:
            ax = 1
        if ax == 0:
            loci = random.randint(r0 + 3, r1 - 2)
            newobj = [p for p in cells if p[0] >= loci]
        else:
            locj = random.randint(c0 + 3, c1 - 2)
            newobj = [p for p in cells if p[1] >= locj]
        for (y, x) in newobj:
            gi[y, x] = ccols[k]
    objs = sorted(_objects(gi), key=lambda o: len(o[1]))
    infos = []
    for col, cells in objs:
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        infos.append((col, min(rs), max(rs), min(cs), max(cs)))
    allowances = [max(1, ((r1 - r0 - 1) * (c1 - c0 - 1)) // 2) for _, r0, r1, c0, c1 in infos]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [random.randint(0, min(meann, a)) for a in allowances]
    while max(chosens) == 0:
        chosens = [random.randint(0, min(meann, a)) for a in allowances]
    mx = max(chosens)
    fixinds = [i for i, cnt in enumerate(chosens) if cnt == mx]
    gogocol = infos[fixinds[0]][0]
    for i in fixinds[1:]:
        chosens[i] -= 1
    for (col, r0, r1, c0, c1), cnt in zip(infos, chosens):
        locs = [(y, x) for y in range(r0 + 1, r1) for x in range(c0 + 1, c1)]
        if cnt > len(locs):
            cnt = len(locs)
        for (y, x) in random.sample(locs, cnt):
            gi[y, x] = noisec
    go = np.full((1, 1), gogocol, dtype=int)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    objs = _objects(I)
    colors = [c for c, _ in objs]
    if len(colors) == len(set(colors)):
        cnt = Counter(I.flatten().tolist())
        noisec = min(cnt, key=lambda k: cnt[k])
    else:
        noisec = Counter(colors).most_common(1)[0][0]
    best = None
    best_n = -1
    for col, cells in objs:
        if col == noisec:
            continue
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        sub = I[min(rs):max(rs) + 1, min(cs):max(cs) + 1]
        n = int((sub == noisec).sum())
        if n > best_n:
            best_n = n
            best = (col, cells)
    col, cells = best
    # pick a cell of the winning region (its own colour) and crop the canvas to it
    r, c = sorted(cells)[0]
    ops = [33, 34]
    sels = [sel_of([(r, c)]), [0, 0, 0, 0]]
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
