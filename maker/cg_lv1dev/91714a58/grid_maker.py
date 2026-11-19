"""
ARC Task: 91714a58 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, targc = random.sample(cols, 2)
    return {"bgc": bgc, "targc": targc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, targc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = list(range(10))
    if bgc is None or targc is None:
        bgc, targc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    remcols = [c for c in cols if c != bgc]
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    gi = [[bgc] * w for _ in range(h)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    noise = random.sample(inds, nnoise)
    ih = random.randint(2, h // 2)
    iw = random.randint(2, w // 2)
    loci = random.randint(0, h - ih)
    locj = random.randint(0, w - iw)
    go = [[bgc] * w for _ in range(h)]
    for i in range(loci, loci + ih):
        for j in range(locj, locj + iw):
            go[i][j] = targc
    for (i, j) in noise:
        gi[i][j] = random.choice(remcols)
    for i in range(loci, loci + ih):
        for j in range(locj, locj + iw):
            gi[i][j] = targc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the canvas colour (majority of I)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    G = I.copy()

    # Reading: a cell of the real block always has a same-colour twin beside it
    # horizontally AND vertically. Noise is thinned in two passes, then any
    # surviving clump that is not the single solid rectangle is erased.

    # Pass 1: erase every non-background cell with no same-colour left/right neighbour
    p1 = []
    for r in range(hi):
        for c in range(wi):
            v = G[r, c]
            if v == bgc:
                continue
            if (c > 0 and G[r, c - 1] == v) or (c + 1 < wi and G[r, c + 1] == v):
                continue
            p1.append((r, c))
    if p1:
        ops.append(int(bgc)); sels.append(sel_of(p1))
        for r, c in p1:
            G[r, c] = bgc

    # Pass 2 (on the thinned grid): erase cells with no same-colour up/down neighbour
    p2 = []
    for r in range(hi):
        for c in range(wi):
            v = G[r, c]
            if v == bgc:
                continue
            if (r > 0 and G[r - 1, c] == v) or (r + 1 < hi and G[r + 1, c] == v):
                continue
            p2.append((r, c))
    if p2:
        ops.append(int(bgc)); sels.append(sel_of(p2))
        for r, c in p2:
            G[r, c] = bgc

    # Pass 3: the kept object is the largest solid single-colour rectangle among survivors
    best = None
    for col in sorted(set(G.flatten().tolist()) - {bgc}):
        M = (G == col)
        heights = [0] * wi
        for r in range(hi):
            for c in range(wi):
                heights[c] = heights[c] + 1 if M[r, c] else 0
            for c1 in range(wi):
                minh = 1 << 30
                for c2 in range(c1, wi):
                    if heights[c2] < minh:
                        minh = heights[c2]
                    if minh < 2:
                        break
                    rw = c2 - c1 + 1
                    if rw < 2:
                        continue
                    area = minh * rw
                    if best is None or area > best[0]:
                        best = (area, r - minh + 1, c1, minh, rw, col)
    keep = set()
    if best is not None:
        _, r0, c0, rh, rw, _ = best
        keep = {(r, c) for r in range(r0, r0 + rh) for c in range(c0, c0 + rw)}

    rest = {(r, c) for r in range(hi) for c in range(wi)
            if G[r, c] != bgc and (r, c) not in keep}
    seen = set()
    for r in range(hi):
        for c in range(wi):
            if (r, c) in rest and (r, c) not in seen:
                col = G[r, c]
                q = deque([(r, c)])
                seen.add((r, c))
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if (ny, nx) in rest and (ny, nx) not in seen and G[ny, nx] == col:
                            seen.add((ny, nx))
                            q.append((ny, nx))
                ops.append(int(bgc)); sels.append(sel_of(cells))
                for y, x in cells:
                    G[y, x] = bgc

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
                        f"num_examples+1 ({num_examples + 1}) for task 91714a58"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 91714a58"
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
                                f"for task 91714a58"
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
                    f"Failed to build a complete episode for task 91714a58 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"91714a58-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
