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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, dotc)]
    numc = random.randint(1, 8)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "dotc": dotc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, dotc=None, ccols=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None or dotc is None:
        bgc, dotc = random.sample(cols, 2)
    if not ccols:
        remcols = [c for c in cols if c not in (bgc, dotc)]
        ccols = random.sample(remcols, random.randint(1, 8))

    def unifint(lb, ub):
        if ub < lb:
            ub = lb
        a = lb + (ub - lb) * diff_lb
        b = lb + (ub - lb) * diff_ub
        return random.randint(int(round(a)), max(int(round(a)), int(round(b))))

    def nbrs(p):
        r, c = p
        return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}

    def shape_of(cells):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    h = unifint(4, max(4, max_h))
    w = unifint(4, max(4, max_w))
    oh = unifint(3, h)
    ow = unifint(3, w)
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = {(loci + 1 + i, locj + 1 + j) for i in range(oh - 2) for j in range(ow - 2)}
    gipro = [[bgc] * w for _ in range(h)]
    for (r, c) in crns:
        gipro[r][c] = dotc
    sp = random.choice(sorted(cands))
    obj = {sp}
    cands.discard(sp)
    ncells = unifint(oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2))
    for _ in range(ncells - 1):
        opts = sorted((cands - obj) & set().union(*[nbrs(p) for p in obj]))
        if not opts:
            break
        obj.add(random.choice(opts))
    while shape_of(obj) != (oh - 2, ow - 2):
        opts = sorted((cands - obj) & set().union(*[nbrs(p) for p in obj]))
        obj.add(random.choice(opts))
    objc = {p: random.choice(ccols) for p in obj}

    go = [row[:] for row in gipro]
    for (r, c), v in objc.items():
        go[r][c] = v

    nperts = unifint(1, max(h, w))
    k = 0
    cur = dict(objc)
    full = {(r, c) for r in range(h) for c in range(w)}
    while min(cur)[0] == loci + 1 and min(p[1] for p in cur) == locj + 1 or k < nperts:
        k += 1
        options = []
        for d in nbrs((0, 0)):
            sh = {(r + d[0], c + d[1]) for (r, c) in cur}
            if not (sh & crns) and sh <= full:
                options.append(d)
        d = random.choice(sorted(options))
        cur = {(r + d[0], c + d[1]): v for (r, c), v in cur.items()}
        if k > 1000:
            break
    gi = [row[:] for row in gipro]
    for (r, c), v in cur.items():
        gi[r][c] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background is fixed per episode: count over every example input plus I
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = ex[0] if not isinstance(ex, dict) else ex["input"]
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    cellmap = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))

    # frame marker: colour whose cells are exactly the 4 corners of its bbox (largest such)
    marker_cells = set()
    best_area = -1
    m_r0 = m_c0 = 0
    for v, cells in cellmap.items():
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if set(cells) == corners:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area > best_area:
                best_area = area
                marker_cells = set(cells)
                m_r0, m_c0 = r0, c0

    # the object = all non-background, non-marker cells, moved as one rigid unit
    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) != bgc and (r, c) not in marker_cells]
    if not obj:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    o_r0 = min(p[0] for p in obj)
    o_c0 = min(p[1] for p in obj)
    dr = (m_r0 + 1) - o_r0
    dc = (m_c0 + 1) - o_c0

    # one grab, then empty selections continue the same object
    cur = list(obj)
    grabbed = False
    v_op = 21 if dr > 0 else 20
    for _ in range(abs(dr)):
        ops.append(v_op)
        sels.append(sel_of(cur) if not grabbed else sel_of([]))
        grabbed = True
        cur = [(r + (1 if dr > 0 else -1), c) for r, c in cur]
    h_op = 22 if dc > 0 else 23
    for _ in range(abs(dc)):
        ops.append(h_op)
        sels.append(sel_of(cur) if not grabbed else sel_of([]))
        grabbed = True
        cur = [(r, c + (1 if dc > 0 else -1)) for r, c in cur]

    src = set(obj)
    dst = set(cur)
    hole = sorted(src - dst)
    if bgc != 0 and hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))

    # 0-coloured object cells are transparent to the object buffer: paint them at destination
    zero_dst = sorted({(r + dr, c + dc) for (r, c) in obj if int(I[r, c]) == 0} - src)
    if zero_dst:
        ops.append(0); sels.append(sel_of(zero_dst))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
