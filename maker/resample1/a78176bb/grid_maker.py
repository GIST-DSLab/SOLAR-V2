"""
ARC Task: a78176bb (RE-ARC) — LLM-generated grid_maker
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

try:
    from utils import unifint
except Exception:
    def unifint(diff_lb, diff_ub, bounds):
        a, b = bounds
        d = random.uniform(diff_lb, diff_ub)
        return min(b, max(a, round(a + (b - a) * d)))


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    variants = [{"mirror": False}, {"mirror": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def _nbrs(cells):
    out = set()
    for r, c in cells:
        out |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
    return out


def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])
    remcols = [c for c in range(10) if c != bgc]
    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    nlns = unifint(diff_lb, diff_ub, (1, max(1, (h + w) // 8)))
    fullinds = {(i, j) for i in range(h) for j in range(w)}
    inds = set(fullinds)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    spopts = [(idx, 0) for idx in range(h - 5, -1, -1)] + [(0, idx) for idx in range(1, w - 4)]
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    ntri = 0
    while succ < nlns and tr < maxtr:
        tr += 1
        sp = random.choice(spopts)
        ln = {(sp[0] + k, sp[1] + k) for k in range(max(h, w))} & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln)
        trid1 = random.randint(2, min(5, len(lno) - 3))
        trid2 = random.randint(2, min(5, len(lno) - 3))
        tri1 = {(i, j) for i in range(trid1) for j in range(trid1) if j >= i}
        a = random.choice(lno[1:-trid1 - 1])
        tri1 = {(i + a[0], j + a[1] + 1) for i, j in tri1}
        tri2 = {(i, j) for i in range(trid2) for j in range(trid2) if j <= i}
        b = random.choice(lno[1:-trid2 - 1])
        tri2 = {(i + b[0] + 1, j + b[1]) for i, j in tri2}
        nex2 = {(sp[0] + k, sp[1] - trid2 - 2 + k) for k in range(max(h, w))} & fullinds
        nex1 = {(sp[0] - trid1 - 2 + k, sp[1] + k) for k in range(max(h, w))} & fullinds
        opts = [(tri1, nex1), (tri2, nex2)]
        random.shuffle(opts)
        inds0 = set(inds)
        placed = set()
        lncol = random.choice(remcols)
        drew = False
        for tri, nex in opts:
            if random.random() < 0.25 and drew:
                continue
            extra = tri | nex
            if not (ln | extra).issubset(inds0):
                continue
            if extra & (placed | _nbrs(placed)):
                continue
            tricol = random.choice([c for c in remcols if c != lncol])
            placed |= extra
            drew = True
            ntri += 1
            for (r, c) in tri:
                gi[r, c] = tricol
            for (r, c) in nex:
                go[r, c] = lncol
        if drew:
            succ += 1
            allobj = ln | placed
            inds = (inds - allobj) - _nbrs(allobj)
            for (r, c) in ln:
                gi[r, c] = lncol
                go[r, c] = lncol
    if ntri == 0:
        raise ValueError("no triangle placed")
    if mirror:
        gi = gi[::-1, :].copy()
        go = go[::-1, :].copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def _components4(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = the canvas colour the generator paints first (dominant colour)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # triangles = 4-connected single-colour blobs of size > 1 (diagonal line cells are 4-isolated)
    tris = [cells for cells in _components4(I, bgc) if len(cells) > 1]
    ops, sels = [], []

    def is_norm_tri(cells, flip):
        cs = [((hi - 1 - r) if flip else r, c) for r, c in cells]
        r0 = min(r for r, _ in cs); c0 = min(c for _, c in cs)
        d = max(r for r, _ in cs) - r0 + 1
        rel = {(r - r0, c - c0) for r, c in cs}
        up = {(i, j) for i in range(d) for j in range(d) if j >= i}
        lo = {(i, j) for i in range(d) for j in range(d) if j <= i}
        return rel == up or rel == lo

    flip = bool(tris) and not is_norm_tri(tris[0], False)
    tf = (lambda r: hi - 1 - r) if flip else (lambda r: r)

    items = []
    for cells in tris:
        cs = {(tf(r), c) for r, c in cells}
        minr = min(r for r, _ in cs); maxr = max(r for r, _ in cs)
        minc = min(c for _, c in cs); maxc = max(c for _, c in cs)
        upper = (minr, maxc) in cs
        if upper:
            lcell = (minr + 1, minc)          # line cell touching the triangle
            anchor = (minr - 1, maxc + 1)     # first cell of the new parallel line
        else:
            lcell = (minr, minc + 1)
            anchor = (maxr + 1, minc - 1)
        lncol = int(I[tf(lcell[0]), lcell[1]])
        diag_line = lcell[1] - lcell[0]
        k = anchor[1] - anchor[0]
        newline = [(tf(r), r + k) for r in range(hi) if 0 <= r + k < wi]
        items.append((diag_line, minr, cells, lncol, newline))
    items.sort(key=lambda t: (t[0], t[1]))
    for _, _, cells, lncol, newline in items:
        # the triangle is absorbed: erase it to background
        ops.append(bgc); sels.append(sel_of(cells))
        # and a parallel line in the line's colour is cast just beyond it
        if newline:
            ops.append(lncol); sels.append(sel_of(newline))
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
                        f"num_examples+1 ({num_examples + 1}) for task a78176bb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a78176bb"
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
                                f"for task a78176bb"
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
                    f"Failed to build a complete episode for task a78176bb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a78176bb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
