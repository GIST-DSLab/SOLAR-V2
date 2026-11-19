"""
ARC Task: e509e548 (RE-ARC) — LLM-generated grid_maker
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
    # Colors 1, 2 and 6 are reserved for the output labels.
    cols = [c for c in range(10) if c not in (1, 2, 6)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d))))


def _connect(a, b):
    (i0, j0), (i1, j1) = a, b
    if i0 == i1:
        return {(i0, j) for j in range(min(j0, j1), max(j0, j1) + 1)}
    if j0 == j1:
        return {(i, j0) for i in range(min(i0, i1), max(i0, i1) + 1)}
    return set()


def _normalize(obj):
    mi = min(i for i, _ in obj)
    mj = min(j for _, j in obj)
    return {(i - mi, j - mj) for i, j in obj}


def _shape(obj):
    return (max(i for i, _ in obj) - min(i for i, _ in obj) + 1,
            max(j for _, j in obj) - min(j for _, j in obj) + 1)


def _getL(h, w):
    return _connect((0, 0), (h - 1, 0)) | _connect((0, 0), (0, w - 1))


def _getU(h, w):
    return (_connect((0, 0), (0, w - 1))
            | _connect((0, 0), (random.randint(1, h - 1), 0))
            | _connect((0, w - 1), (random.randint(1, h - 1), w - 1)))


def _getH(h, w):
    off = random.randint(1, w - 2)
    leg = _connect((0, 0), (h - 1, 0)) | _connect((h - 1, 0), (h - 1, random.randint(1, w - 1)))
    return _connect((0, 0), (0, w - 1)) | {(i, j + off) for i, j in leg}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 6)]
    if bgc is None:
        bgc = random.choice(cols)
    pairs = ((2, 2, _getL, 1), (2, 3, _getU, 6), (3, 3, _getH, 2))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in cols if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (1, 6))
    ccols = random.sample(remcols, ncols)
    nobjs = _unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    mfs = [
        lambda o: set(o),
        lambda o: {(j, i) for i, j in o},          # dmirror
        lambda o: {(-j, -i) for i, j in o},        # cmirror
        lambda o: {(i, -j) for i, j in o},         # vmirror
        lambda o: {(-i, j) for i, j in o},         # hmirror
    ]
    while succ < nobjs and tr < maxtr:
        tr += 1
        minh, minw, getter, outc = random.choice(pairs)
        oh = random.randint(minh, 6)
        ow = random.randint(minw, 6)
        obj = getter(oh, ow)
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = _normalize(fn(obj))
        oh, ow = _shape(obj)
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(cands)
        plcd = {(i + li, j + lj) for i, j in obj}
        if plcd.issubset(inds):
            succ += 1
            nb = set()
            for i, j in plcd:
                nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
            inds = (inds - plcd) - nb
            col = random.choice(ccols)
            for i, j in plcd:
                gi[i][j] = col
                go[i][j] = outc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared by the verifier and the code-golf entry): first collapse
    # every object to one uniform colour (the default label 2 = "H"), then pick
    # out the special shapes by structure and relabel them: L-shapes (a single
    # corner path, size == h+w-1) become 1, and U-shapes (every cell on the
    # bbox border) become 6.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the canvas colour shared by the demonstrations
    bgc = None
    if examples:
        common = None
        for ex_in, _ in examples:
            s = set(np.asarray(ex_in, dtype=int).flatten().tolist())
            common = s if common is None else common & s
        common = (common or set()) & set(I.flatten().tolist())
        if len(common) == 1:
            bgc = next(iter(common))
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # find the objects (4-connected, same colour) in I
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            stack = [(r, c)]
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            objs.append(cells)

    # Step 1: paint all foreground cells (every non-background cell of I) with 2
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if any(I[r, c] != 2 for r, c in fg):
        ops.append(2); sels.append(sel_of(fg))

    # Step 2: relabel each object by its shape class
    for cells in objs:
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        if len(cells) == oh + ow - 1:
            new = 1
        elif all(p[0] in (r0, r1) or p[1] in (c0, c1) for p in cells):
            new = 6
        else:
            continue  # H-shape: already labelled 2
        sr, sc = cells[0]
        ops.append(10 + new); sels.append(sel_of([(sr, sc)]))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e509e548"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e509e548"
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
                                f"for task e509e548"
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
                    f"Failed to build a complete episode for task e509e548 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e509e548-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
