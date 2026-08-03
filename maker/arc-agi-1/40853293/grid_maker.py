"""
ARC Task: 40853293 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random
from collections import Counter, defaultdict


def sample_colors(num_examples=None) -> dict:
    # Rule depends only on presence/pattern of dot-pairs, not their specific colors.
    # Only background must be fixed per episode.
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(dlb, dub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + round((b - a) * dlb)
        hi = a + round((b - a) * dub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    def connect(a, b):
        (r0, c0), (r1, c1) = a, b
        cells = set()
        if r0 == r1:
            for c in range(min(c0, c1), max(c0, c1) + 1):
                cells.add((r0, c))
        elif c0 == c1:
            for r in range(min(r0, r1), max(r0, r1) + 1):
                cells.add((r, c0))
        return cells

    cols = list(range(0, 10))
    max_h = max(5, min(30, max_h))
    max_w = max(5, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))

    nlines = unifint(diff_lb, diff_ub, (2, max(2, min(8, (h * w) // 2))))
    if nlines < 2:
        nlines = 2
    nhorilines = random.randint(1, nlines - 1)
    nvertilines = nlines - nhorilines

    ilocs = list(range(0, h))
    ilocs = random.sample(ilocs, min(nhorilines, len(ilocs)))

    remcols = [c for c in cols if c != bgc]

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    go = [[bgc for _ in range(w)] for _ in range(h)]

    # horizontal lines
    for ii in ilocs:
        if not remcols:
            break
        llen = unifint(diff_lb, diff_ub, (2, max(2, w - 1)))
        llen = min(llen, w)
        js = random.randint(0, w - llen)
        je = js + llen - 1
        a = (ii, js)
        b = (ii, je)
        hln = connect(a, b)
        col = random.choice(remcols)
        remcols.remove(col)
        gi[a[0]][a[1]] = col
        gi[b[0]][b[1]] = col
        for (r, c) in hln:
            go[r][c] = col

    # vertical lines (operate on columns; only columns with >1 free bg cell)
    jlocs = []
    for j in range(w):
        colcells = [gi[r][j] for r in range(h)]
        if sum(1 for e in colcells if e == bgc) > 1:
            jlocs.append(j)
    nvertilines = min(nvertilines, len(jlocs))
    if nvertilines > 0:
        jlocs = random.sample(jlocs, nvertilines)
        for jj in jlocs:
            if not remcols:
                break
            jcands = [idx for idx in range(h) if gi[idx][jj] == bgc]
            kk = len(jcands)
            if kk < 2:
                continue
            llen = unifint(diff_lb, diff_ub, (2, kk))
            llen = min(llen, kk)
            sp = random.randint(0, kk - llen)
            ep = sp + llen - 1
            sp = jcands[sp]
            ep = jcands[ep]
            a = (sp, jj)
            b = (ep, jj)
            vln = connect(a, b)
            col = random.choice(remcols)
            remcols.remove(col)
            gi[a[0]][a[1]] = col
            gi[b[0]][b[1]] = col
            for (r, c) in vln:
                go[r][c] = col

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cells = defaultdict(list)
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells[v].append((r, c))

    horiz = []  # (row, c0, c1, color) -> full 1xN rectangle span
    vert = []   # (r0, r1, col_index, color) -> full Nx1 rectangle span
    for col, pts in cells.items():
        if len(pts) < 2:
            continue
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        if len(set(rs)) == 1:            # dots share a row -> horizontal line
            r = rs[0]
            horiz.append((r, min(cs), max(cs), col))
        elif len(set(cs)) == 1:          # dots share a column -> vertical line
            c = cs[0]
            vert.append((min(rs), max(rs), c, col))

    ops, sels = [], []

    # Paint horizontal lines first ...
    for r, c0, c1, col in horiz:
        # exact full 1xN rectangle -> bbox selection is the intended cell set
        ops.append(int(col))
        sels.append([r, c0, 0, c1 - c0])

    # ... then vertical lines on top (they win at crossings, matching verifier order)
    for r0, r1, c, col in vert:
        # exact full Nx1 rectangle -> bbox selection is the intended cell set
        ops.append(int(col))
        sels.append([r0, c, r1 - r0, 0])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 40853293"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 40853293"
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
                                f"for task 40853293"
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
                    f"Failed to build a complete episode for task 40853293 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"40853293-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
