"""
ARC Task: f25fbde4 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))

    h = unifint(diff_lb, diff_ub, (2, max_h))
    w = unifint(diff_lb, diff_ub, (2, max_w))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (min(15, h - 1) * min(15, w - 1)) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    bh, bw = min(15, h - 1), min(15, w - 1)
    bounds = {(i, j) for i in range(bh) for j in range(bw)}
    ch = random.choice(sorted(bounds))
    shp = {ch}
    bounds.discard(ch)
    for _ in range(nc):
        cand = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    cand.add((i + di, j + dj))
        cand = (bounds - shp) & cand
        if not cand:
            break
        shp.add(random.choice(sorted(cand)))
    mi = min(i for i, _ in shp)
    mj = min(j for _, j in shp)
    shp = {(i - mi, j - mj) for i, j in shp}
    oh = max(i for i, _ in shp) + 1
    ow = max(j for _, j in shp) + 1
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = [[bgc] * w for _ in range(h)]
    for i, j in shp:
        gi[i + loci][j + locj] = fgc
    sub = [row[locj:locj + ow] for row in gi[loci:loci + oh]]
    go = []
    for row in sub:
        dr = [v for v in row for _ in range(2)]
        go.append(list(dr))
        go.append(list(dr))
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: crop to the shape's bounding box, then rebuild it row by row --
    # each source row is stretched horizontally (every cell -> 2 cells) and laid
    # down twice (a 2-row strip).  Rows are processed bottom-up so each source
    # row is still intact when its strip is painted.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour, a strict majority by construction
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    rows = [r for r in range(hi) if (I[r] != bgc).any()]
    cols = [c for c in range(wi) if (I[:, c] != bgc).any()]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    h, w = r1 - r0 + 1, c1 - c0 + 1

    ops, sels = [], []
    # (1) crop canvas to the shape's bounding box (full rectangle)
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
    # (2) expand canvas to the doubled size (full rectangle)
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])

    G = np.zeros((2 * h, 2 * w), dtype=int)
    G[:h, :w] = I[r0:r1 + 1, c0:c1 + 1]

    # (3) row by row, bottom-up: paint the 2-row strip for source row i,
    #     one colour at a time (the doubled cells of that colour)
    for i in range(h - 1, -1, -1):
        src = [int(v) for v in G[i, :w]]
        order = []
        for v in src:
            if v != bgc and v not in order:
                order.append(v)
        order.append(bgc)
        for col in order:
            cells = [(2 * i + a, 2 * j + b) for j in range(w) if src[j] == col
                     for a in (0, 1) for b in (0, 1)]
            cells = [(r, c) for r, c in cells if G[r, c] != col]  # already that colour now
            if not cells:
                continue
            ops.append(col); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = col

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f25fbde4"
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
                                f"for task f25fbde4"
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
                    f"Failed to build a complete episode for task f25fbde4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f25fbde4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
