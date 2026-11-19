"""
ARC Task: 025d127b (RE-ARC) — LLM-generated grid_maker
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
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))
    if bgc is None:
        bgc = random.randint(0, 9)
    if not ccols:
        rem = [c for c in range(10) if c != bgc]
        ccols = random.sample(rem, random.randint(1, 9))
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def connect(a, b):
        (r0, c0), (r1, c1) = a, b
        if r0 == r1:
            return {(r0, c) for c in range(min(c0, c1), max(c0, c1) + 1)}
        if c0 == c1:
            return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
        if r1 - r0 == c1 - c0:
            return {(r0 + k, c0 + k) for k in range(abs(r1 - r0) + 1)} if r1 >= r0 else \
                   {(r1 + k, c1 + k) for k in range(abs(r1 - r0) + 1)}
        return set()

    def nbrs(cell):
        r, c = cell
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 6)
        ow = random.randint(3, 6)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loc = random.choice(cands)
        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        outobj = {(r, c + 1) for r, c in topl} | botl | {(r, c + 1) for r, c in leftl} | \
            connect((1, ow + 1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)}
        rm = max(c for _, c in inobj)
        outobj = {ij for ij in outobj if ij[1] <= rm}
        full = inobj | outobj
        sh = lambda s: {(r + loc[0], c + loc[1]) for r, c in s}
        inobj, outobj, full = sh(inobj), sh(outobj), sh(full)
        if full.issubset(inds):
            nb = set()
            for x in full:
                nb |= nbrs(x)
            inds = (inds - full) - nb
            succ += 1
            col = random.choice(ccols)
            for r, c in inobj:
                gi[r][c] = col
            for r, c in outobj:
                go[r][c] = col
    return {"input": gi, "output": go}


def _components(I, bgc, diag):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        if (da, db) == (0, 0) or (not diag and da and db):
                            continue
                        x, y = a + da, b + db
                        if 0 <= x < H and 0 <= y < W and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True
                            stack.append((x, y))
            comps.append((int(col), cells))
    return comps


def _sub4(cells):
    cs = set(cells)
    parts, seen = [], set()
    for s in sorted(cs):
        if s in seen:
            continue
        seen.add(s)
        st, part = [s], []
        while st:
            a, b = st.pop()
            part.append((a, b))
            for x, y in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if (x, y) in cs and (x, y) not in seen:
                    seen.add((x, y))
                    st.append((x, y))
        parts.append(part)
    return parts


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    objs = _components(I, bgc, diag=True)
    objs.sort(key=lambda o: (min(r for r, _ in o[1]), min(c for _, c in o[1])))
    for col, cells in objs:
        parts = _sub4(cells)
        # anchored part: the bottom edge plus the cell sitting on its right end
        # (a 4-part whose rightmost column holds two vertically stacked cells)
        anchor = set()
        for p in parts:
            mc = max(c for _, c in p)
            rows = {r for r, c in p if c == mc}
            if any(r - 1 in rows for r in rows):
                anchor |= set(p)
        moving = [x for x in cells if x not in anchor]
        if not moving:
            continue
        dest = [(r, c + 1) for r, c in moving]
        hole = sorted(set(moving) - set(dest))
        if col != 0:
            ops.append(22); sels.append(sel_of(moving))   # MoveR 1 on the non-anchored part
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
        else:
            # ARCLE cannot carry colour-0 cells in a Move; repaint instead
            new = sorted(set(dest) - set(moving))
            if hole:
                ops.append(bgc); sels.append(sel_of(hole))
            if new:
                ops.append(0); sels.append(sel_of(new))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 025d127b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 025d127b"
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
                                f"for task 025d127b"
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
                    f"Failed to build a complete episode for task 025d127b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"025d127b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
