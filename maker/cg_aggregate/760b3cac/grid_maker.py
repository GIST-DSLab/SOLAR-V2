"""
ARC Task: 760b3cac (RE-ARC) — LLM-generated grid_maker
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

_OBJL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
VARIANTS = [{"sgn": s, "rot": k} for s in (-1, 1) for k in range(4)]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, indc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "indc": indc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, indc=2,
             sgn=None, rot=None) -> dict:
    def unif(lo, hi):
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))
    if sgn is None:
        sgn = random.choice((-1, 1))
    if rot is None:
        rot = random.randint(0, 3)
    lim = min(max_h, max_w)
    h = unif(5, min(30, lim))
    w = unif(3, max(3, min(14, (lim - 1) // 2)))
    w = 2 * w + 1
    objh = unif(1, h - 3)
    objw = 2 * unif(1, w // 6) + 1
    gi = np.full((h, w), bgc, dtype=int)
    obj = _OBJL if sgn == -1 else frozenset((r, 2 - c) for r, c in _OBJL)
    for r, c in obj:
        gi[h - 3 + r, w // 2 - 1 + c] = indc
    inds = {(i, j) for i in range(objh) for j in range(objw)}
    sp = random.choice(sorted(inds))
    objx = {sp}

    def nbrs(s):
        out = set()
        for (i, j) in s:
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    if a or b:
                        out.add((i + a, j + b))
        return out
    numcd = unif(0, (objh * objw) // 2)
    numc = random.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for _ in range(numc - 1):
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    while max(j for _, j in objx) - min(j for _, j in objx) + 1 != objw:
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    mi, mj = min(i for i, _ in objx), min(j for _, j in objx)
    objx = {(i - mi, j - mj) for i, j in objx}
    oh = max(i for i, _ in objx) + 1
    ow = max(j for _, j in objx) + 1
    loci = random.randint(0, h - 3 - oh)
    locj = w // 2 - ow // 2
    plcd = {(i + loci, j + locj) for i, j in objx}
    for r, c in plcd:
        gi[r, c] = objc
    go = gi.copy()
    for r, c in plcd:
        nc = (2 * locj + ow - 1 - c) + ow * sgn
        if 0 <= nc < w:
            go[r, nc] = objc
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _edge_direction(arrow):
    r0, r1, c0, c1 = _bbox(arrow)
    out = []
    if sum(1 for r, _ in arrow if r == r1) == 2: out.append((1, 0))
    if sum(1 for r, _ in arrow if r == r0) == 2: out.append((-1, 0))
    if sum(1 for _, c in arrow if c == c1) == 2: out.append((0, 1))
    if sum(1 for _, c in arrow if c == c0) == 2: out.append((0, -1))
    return out


def _is_arrow_like(cells):
    if len(cells) != 5:
        return False
    r0, r1, c0, c1 = _bbox(cells)
    if r1 - r0 != 2 or c1 - c0 != 2:
        return False
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    border = {(r, c) for (r, c) in cells if r in (r0, r1) or c in (c0, c1)}
    return len(border) == 4 and len(set(cells) & corners) == 1


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # background: the colour every demonstration input is drawn on
    cnt = Counter(I.flatten().tolist())
    for ex_i, _ in (examples or []):
        cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, set()).add((r, c))
    if not groups:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the reflected object's colour is the one the demonstrations grow
    main_col = None
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int)
        ex_o = np.asarray(ex_o, dtype=int)
        if ex_i.shape != ex_o.shape:
            continue
        for v in np.unique(ex_o).tolist():
            if int((ex_o == v).sum()) > int((ex_i == v).sum()):
                grown[int(v)] += 1
    for v, _ in grown.most_common():
        if v in groups:
            main_col = v
            break
    if main_col is None:
        arrows = [k for k in groups if _is_arrow_like(groups[k])]
        rest = [k for k in groups if k not in arrows]
        main_col = max(rest or list(groups), key=lambda k: len(groups[k]))
    others = [k for k in groups if k != main_col]
    main = groups[main_col]
    arrow = groups[others[0]] if others else set()
    d = (_edge_direction(arrow) if arrow else []) or [(1, 0)]
    d = d[0]

    # Reflect the object across its bbox edge on the arrow's side (the side of
    # the arrow's 3x3 box holding two cells). m = how many rows/cols of the
    # reflection land on the canvas; those are the reflection of the m source
    # lines nearest that edge — that strip is the unit copied.
    r0, r1, c0, c1 = _bbox(main)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    if d == (0, -1):
        m = min(ow, c0)
        src = [r0, c0, oh - 1, m - 1]; dst = [r0, c0 - m, oh - 1, m - 1]
    elif d == (0, 1):
        m = min(ow, wi - 1 - c1)
        src = [r0, c1 - m + 1, oh - 1, m - 1]; dst = [r0, c1 + 1, oh - 1, m - 1]
    elif d == (-1, 0):
        m = min(oh, r0)
        src = [r0, c0, m - 1, ow - 1]; dst = [r0 - m, c0, m - 1, ow - 1]
    else:
        m = min(oh, hi - 1 - r1)
        src = [r1 - m + 1, c0, m - 1, ow - 1]; dst = [r1 + 1, c0, m - 1, ow - 1]
    if m <= 0:
        return [34], [[0, 0, hi - 1, wi - 1]]

    sr, sc = src[0], src[1]
    dr_, dc_ = dst[0], dst[1]
    sh, sw = src[2] + 1, src[3] + 1
    strip = {(r, c) for (r, c) in main if sr <= r < sr + sh and sc <= c < sc + sw}
    if d[1] == 0:
        mirrored = {(dr_ + (sh - 1 - (r - sr)), dc_ + (c - sc)) for r, c in strip}
    else:
        mirrored = {(dr_ + (r - sr), dc_ + (sw - 1 - (c - sc))) for r, c in strip}

    if main_col != 0:
        # Full rectangles on purpose (background included): the strip of the
        # object next to its reflecting edge is copied, pasted just past that
        # edge, then mirrored in place across that same edge.
        ops.append(28); sels.append(src)                 # CopyI source strip
        ops.append(30); sels.append([dr_, dc_, 0, 0])    # Paste beyond the edge
        pasted = {(dr_ + r - sr, dc_ + c - sc) for r, c in strip}
        if pasted != mirrored:                           # symmetric strip: no flip
            ops.append(27 if d[1] == 0 else 26); sels.append(dst)
    else:
        # colour 0 is transparent to Copy/Paste: draw the reflection directly
        ops.append(0); sels.append(sel_of(sorted(mirrored)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
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
                                f"for task 760b3cac"
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
                    f"Failed to build a complete episode for task 760b3cac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"760b3cac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
