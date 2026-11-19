"""
ARC Task: b527c5c6 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    # The rule (extend a rectangle from its marked dot) does not depend on which
    # foreground colours are used, so only the background is fixed per episode.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(list(range(10)))
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    remcols = [c for c in range(10) if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    noccs = _unifint(diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs
    m = min(h, w)
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = random.randint(3, random.randint(3, m // 2 - 1))
        d2 = random.randint(d1 * 2 + 1, random.randint(d1 * 2 + 1, m - 1))
        oh, ow = random.sample([d1, d2], 2)
        cands = [ij for ij in inds if 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        s = min(oh, ow)
        if ow < oh:
            # dot on the left or right column, ray goes horizontally
            rows = list(range(loci + ow - 1, loci + oh - ow + 1))
            if not rows:
                continue
            side = random.choice([-1, 1])
            dr = random.choice(rows)
            dc = locj if side == -1 else locj + ow - 1
            direc = (0, side)
        else:
            cols = list(range(locj + oh - 1, locj + ow - oh + 1))
            if not cols:
                continue
            side = random.choice([-1, 1])
            dc = random.choice(cols)
            dr = loci if side == -1 else loci + oh - 1
            direc = (side, 0)
        ln = set()
        r, c = dr, dc
        while 0 <= r < h and 0 <= c < w:
            ln.add((r, c))
            r += direc[0]
            c += direc[1]
        lrs = [p[0] for p in ln]
        lcs = [p[1] for p in ln]
        shell = set()
        for i in range(min(lrs) - (s - 1), max(lrs) + s):
            for j in range(min(lcs) - (s - 1), max(lcs) + s):
                if 0 <= i < h and 0 <= j < w:
                    shell.add((i, j))
        sqc, dotc = random.sample(ccols, 2)
        sq_cells = (bd | shell) - ln
        goobji = sq_cells | ln
        if goobji.issubset(inds):
            succ += 1
            nb = set()
            for (i, j) in bd:
                for (a, b) in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                    nb.add((a, b))
            inds = (inds - goobji) - nb
            for (i, j) in bd:
                gi[i][j] = sqc
            gi[dr][dc] = dotc
            for (i, j) in sq_cells:
                go[i][j] = sqc
            for (i, j) in ln:
                go[i][j] = dotc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The generator never places a rectangle on the outer border, so the corner is background.
    bgc = int(I[0, 0])

    ops, sels = [], []
    visited = np.zeros((hi, wi), dtype=bool)
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            visited[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not visited[nr, nc] and I[nr, nc] != bgc:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            cnt = Counter(int(I[r, c]) for r, c in cells)
            sqc = cnt.most_common(1)[0][0]
            dots = [(r, c) for r, c in cells if int(I[r, c]) != sqc]
            if len(dots) != 1:
                continue
            dr, dc = dots[0]
            dotc = int(I[dr, dc])
            r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            s = min(oh, ow)

            if ow > oh:
                # wide rectangle: the dot sits on the top or bottom edge, so the rectangle extends vertically
                rows = list(range(0, r0)) if dr == r0 else list(range(r1 + 1, hi))
                cols = list(range(max(0, dc - (s - 1)), min(wi - 1, dc + (s - 1)) + 1))
                ray = [(r, dc) for r in rows]
            else:
                # tall rectangle: the dot sits on the left or right edge, so the rectangle extends horizontally
                cols = list(range(0, c0)) if dc == c0 else list(range(c1 + 1, wi))
                rows = list(range(max(0, dr - (s - 1)), min(hi - 1, dr + (s - 1)) + 1))
                ray = [(dr, c) for c in cols]
            band = [(r, c) for r in rows for c in cols]
            if not band:
                continue
            # 1) extend the rectangle from its marked edge: a band 2s-1 wide, centred on the dot,
            #    running from that edge to the grid border, painted in the rectangle's colour
            ops.append(sqc); sels.append(sel_of(band))
            # 2) shoot the dot's colour from the dot to the grid border, drawn over the band
            ops.append(dotc); sels.append(sel_of(ray))

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
                        f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
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
                                f"for task b527c5c6"
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
                    f"Failed to build a complete episode for task b527c5c6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b527c5c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
