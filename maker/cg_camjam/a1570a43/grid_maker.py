"""
ARC Task: a1570a43 (RE-ARC) — LLM-generated grid_maker
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
    bgc, dotc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, dotc)]
    ccols = random.sample(remcols, 8)  # ordered pool; generate() takes a prefix
    return {"bgc": bgc, "dotc": dotc, "ccols": ccols}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    v = random.uniform(lo, hi)
    return max(a, min(b, int(round(v))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, ccols=None) -> dict:
    cols = list(range(10))
    if bgc is None or dotc is None:
        bgc, dotc = random.sample(cols, 2)
    if ccols is None:
        ccols = random.sample([c for c in cols if c not in (bgc, dotc)], 8)
    max_h = max(4, min(30, max_h))
    max_w = max(4, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (4, max_h))
    w = _unifint(diff_lb, diff_ub, (4, max_w))
    oh = _unifint(diff_lb, diff_ub, (3, h))
    ow = _unifint(diff_lb, diff_ub, (3, w))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = {(loci + 1 + i, locj + 1 + j) for i in range(oh - 2) for j in range(ow - 2)}
    numc = _unifint(diff_lb, diff_ub, (1, 8))
    cc = list(ccols)[:numc]

    def nbrs(p):
        r, c = p
        return {(r + a, c + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}

    def shape(s):
        rs = [p[0] for p in s]; cs = [p[1] for p in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    sp = random.choice(sorted(cands))
    obj = {sp}
    ncells = _unifint(diff_lb, diff_ub, (oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2)))
    for _ in range(ncells - 1):
        nb = set()
        for p in obj:
            nb |= nbrs(p)
        opts = sorted((cands - obj) & nb)
        if not opts:
            break
        obj.add(random.choice(opts))
    while shape(obj) != (oh - 2, ow - 2):
        nb = set()
        for p in obj:
            nb |= nbrs(p)
        obj.add(random.choice(sorted((cands - obj) & nb)))
    objc = {p: random.choice(cc) for p in obj}

    base = np.full((h, w), bgc, dtype=int)
    for (r, c) in crns:
        base[r, c] = dotc
    go = base.copy()
    for (r, c), v in objc.items():
        go[r, c] = v

    nperts = _unifint(diff_lb, diff_ub, (1, max(h, w)))
    k = 0
    off = (0, 0)
    target = (loci + 1, locj + 1)
    pts = list(objc.keys())
    ul0 = (min(p[0] for p in pts), min(p[1] for p in pts))
    while (ul0[0] + off[0], ul0[1] + off[1]) == target or k < nperts:
        k += 1
        options = []
        for d in nbrs((0, 0)):
            nd = (off[0] + d[0], off[1] + d[1])
            sh = [(r + nd[0], c + nd[1]) for r, c in pts]
            if all(0 <= r < h and 0 <= c < w for r, c in sh) and not (set(sh) & crns):
                options.append(nd)
        off = random.choice(sorted(options))
    gi = base.copy()
    for (r, c), v in objc.items():
        gi[r + off[0], c + off[1]] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading: the four same-coloured corner dots form a frame; everything else
    # non-background is ONE object that slides (as a unit) until its upper-left
    # sits just inside the frame's upper-left corner. The golf entry reads it the
    # same way (shift the whole object along one axis, then the other), so the
    # route is: one grab, unit Moves along rows, then along columns.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cellmap = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))

    marker_cells = set()
    best_area = -1
    m_r0 = m_c0 = 0
    for v, cells in cellmap.items():
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if set(cells) == corners:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area > best_area:
                best_area = area
                marker_cells = set(cells)
                m_r0, m_c0 = r0, c0

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and (r, c) not in marker_cells]
    if not obj:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    o_r0 = min(p[0] for p in obj)
    o_c0 = min(p[1] for p in obj)
    dr = (m_r0 + 1) - o_r0
    dc = (m_c0 + 1) - o_c0

    cur = list(obj)
    grabbed = False
    sr = 1 if dr > 0 else -1
    for _ in range(abs(dr)):
        ops.append(21 if dr > 0 else 20)
        sels.append(sel_of(cur) if not grabbed else sel_of([]))
        grabbed = True
        cur = [(r + sr, c) for r, c in cur]
    sc = 1 if dc > 0 else -1
    for _ in range(abs(dc)):
        ops.append(22 if dc > 0 else 23)
        sels.append(sel_of(cur) if not grabbed else sel_of([]))
        grabbed = True
        cur = [(r, c + sc) for r, c in cur]

    src = set(obj)
    dst = set(cur)
    hole = sorted(src - dst)
    if bgc != 0 and hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))

    # 0-coloured object cells are transparent in ARCLE's object buffer:
    # paint them at their shifted positions (inside old footprint they already read 0)
    zero_dst = sorted({(r + dr, c + dc) for (r, c) in obj if int(I[r, c]) == 0} - src)
    if zero_dst:
        ops.append(0); sels.append(sel_of(zero_dst))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a1570a43"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a1570a43"
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
                                f"for task a1570a43"
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
                    f"Failed to build a complete episode for task a1570a43 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a1570a43-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
