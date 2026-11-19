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
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        return int(round(a + (b - a) * random.uniform(lb, ub)))

    h = unifint(diff_lb, diff_ub, (2, max(2, max_h)))
    w = unifint(diff_lb, diff_ub, (2, max(2, max_w)))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (min(15, h - 1) * min(15, w - 1)) // 2)))
    nc = min(max(1, ncd), (h - 1) * (w - 1) - 1)
    bh, bw = min(15, h - 1), min(15, w - 1)
    bounds = {(i, j) for i in range(bh) for j in range(bw)}
    nc = min(nc, len(bounds) - 1)
    ch = random.choice(sorted(bounds))
    shp = {ch}
    bounds.discard(ch)
    for _ in range(nc):
        cand = set()
        for (i, j) in shp:
            for (a, b) in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
                if (a, b) in bounds and (a, b) not in shp:
                    cand.add((a, b))
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
    cropped = [row[locj:locj + ow] for row in gi[loci:loci + oh]]
    go = []
    for row in cropped:
        r2 = []
        for v in row:
            r2 += [v, v]
        go.append(list(r2))
        go.append(list(r2))
    return {"input": [tuple(r) for r in gi], "output": [tuple(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    # Reading (line-wise, rows first): the picture is a stack of LINES.
    # Throw away the background-only rows, double every remaining row (each
    # row becomes two); throw away the background-only columns, double every
    # remaining column.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the canvas colour (strict majority); confirm with demos
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        votes = Counter()
        for ex in examples:
            EI = np.asarray(ex[0], dtype=int)
            votes[Counter(EI.flatten().tolist()).most_common(1)[0][0]] += 1
        vb = votes.most_common(1)[0][0]
        if vb in set(I.flatten().tolist()):
            bgc = vb

    rows = [r for r in range(hi) if any(I[r, c] != bgc for c in range(wi))]
    cols = [c for c in range(wi) if any(I[r, c] != bgc for r in range(hi))]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1

    ops, sels = [], []
    # faithful picture of the working canvas (zero-padded 30x30, as in ARCLE)
    G = np.zeros((30, 30), dtype=int)
    G[:hi, :wi] = I
    clip = None  # (list, kind)

    def crop(r, c, hh, ww):
        nonlocal G
        sub = G[r:r + hh, c:c + ww]
        N = np.zeros((30, 30), dtype=int)
        N[:sub.shape[0], :sub.shape[1]] = sub
        G = N
        ops.append(33); sels.append([r, c, hh - 1, ww - 1])

    # (1) drop the background-only rows above the shape: keep the full-width
    #     band starting at row r0 (full rectangle crop). Nothing to drop when
    #     the shape already starts on the top row.
    if r0 > 0:
        crop(r0, 0, h, wi)
    # (2) drop the background-only columns left of the shape and make room for
    #     the doubled rows (height 2h) -- full rectangle. If the shape already
    #     starts in column 0 there is nothing to drop, so this one resize goes
    #     straight to the final 2h x 2w canvas.
    if c0 > 0:
        crop(0, c0, 2 * h, w)
        widened = False
    else:
        crop(0, 0, 2 * h, 2 * w)
        widened = True

    def double_lines(n_lines, length, axis):
        nonlocal clip

        def get(k):
            return [int(v) for v in (G[k, :length] if axis == "row" else G[:length, k])]

        def cell(k, p):
            return (k, p) if axis == "row" else (p, k)

        for k in range(n_lines - 1, -1, -1):
            src = get(k)
            for t in (2 * k, 2 * k + 1):
                cur = get(t)
                if cur == src:
                    continue
                # Paste never writes colour 0 -- clear those cells explicitly
                clear = [cell(t, p) for p in range(length) if src[p] == 0 and cur[p] != 0]
                if clear:
                    ops.append(0); sels.append(sel_of(clear))
                    for (a, b) in clear:
                        G[a, b] = 0
                cur = get(t)
                if cur == src:
                    continue

                def pasted(cl):
                    if cl is None or cl[1] != axis:
                        return None
                    arr = cl[0]
                    return [arr[p] if p < len(arr) and arr[p] != 0 else cur[p]
                            for p in range(length)]

                if pasted(clip) != src:
                    # CopyO the whole line k of the current canvas
                    ops.append(29)
                    sels.append([k, 0, 0, length - 1] if axis == "row"
                                else [0, k, length - 1, 0])
                    clip = (list(src), axis)
                ops.append(30)
                sels.append([t, 0, 0, 0] if axis == "row" else [0, t, 0, 0])
                for p in range(length):
                    if clip[0][p] != 0:
                        a, b = cell(t, p)
                        G[a, b] = clip[0][p]

    # (3) double every row: row k -> rows 2k, 2k+1 (bottom-up so no source
    #     row is overwritten before it is copied)
    double_lines(h, w, "row")

    # (4) widen the canvas to 2w columns (full rectangle)
    if not widened:
        crop(0, 0, 2 * h, 2 * w)

    # (5) double every column: column k -> columns 2k, 2k+1 (right-to-left)
    double_lines(w, 2 * h, "col")

    # Self-check: replay on a model of the canvas (built from I alone) and
    # drop any op whose deletion leaves the final canvas unchanged -- with
    # colour 0 in play a later column paste can make an earlier row op moot.
    def run(o_list, s_list):
        g = np.zeros((30, 30), dtype=int)
        g[:hi, :wi] = I
        dims = (hi, wi)
        cl = np.zeros((0, 0), dtype=int)
        for op, sl in zip(o_list, s_list):
            if isinstance(sl, dict):
                cells = [tuple(x) for x in sl["cells"]]
            else:
                r, c, dh, dw = sl
                cells = [(a, b) for a in range(r, r + dh + 1) for b in range(c, c + dw + 1)]
            rs = [a for a, _ in cells]; cs = [b for _, b in cells]
            ra, rb, ca, cb = min(rs), max(rs), min(cs), max(cs)
            if op == 0:
                for a, b in cells:
                    g[a, b] = 0
            elif op == 33:
                sub = g[ra:rb + 1, ca:cb + 1].copy()
                g = np.zeros((30, 30), dtype=int)
                g[:sub.shape[0], :sub.shape[1]] = sub
                dims = sub.shape
            elif op == 29:
                cl = g[ra:rb + 1, ca:cb + 1].copy()
            elif op == 30:
                for a in range(cl.shape[0]):
                    for b in range(cl.shape[1]):
                        if cl[a, b] != 0 and ra + a < 30 and ca + b < 30:
                            g[ra + a, ca + b] = cl[a, b]
        return dims, g[:dims[0], :dims[1]].copy()

    target = run(ops, sels)
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            o2 = ops[:k] + ops[k + 1:]
            s2 = sels[:k] + sels[k + 1:]
            res = run(o2, s2)
            if res[0] == target[0] and np.array_equal(res[1], target[1]):
                ops, sels = o2, s2
                changed = True
                break

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
