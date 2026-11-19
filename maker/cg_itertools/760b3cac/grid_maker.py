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

_VARIANTS = [{"side": s, "rot": k} for s in ("L", "R") for k in range(4)]


def sample_colors(num_examples=None) -> dict:
    bgc, objc, indc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(_VARIANTS):
        ex = [dict(v) for v in _VARIANTS]
        ex += [dict(random.choice(_VARIANTS)) for _ in range(n_ex - len(_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(_VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "indc": indc, "instance_plan": plan}


def _unifint(lb, ub, bounds):
    a, b = bounds
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, objc=1, indc=2,
             side=None, rot=None, **kw) -> dict:
    if side is None:
        side = random.choice(("L", "R"))
    if rot is None:
        rot = random.randint(0, 3)
    lim = min(max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (5, lim))
    w = _unifint(diff_lb, diff_ub, (3, max(3, (lim - 1) // 2)))
    w = 2 * w + 1
    objh = _unifint(diff_lb, diff_ub, (1, h - 3))
    objw = _unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    objL = {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)}
    if side == "L":
        arrow, sgn = objL, -1
    else:
        arrow, sgn = {(i, 2 - j) for i, j in objL}, 1
    gi = np.full((h, w), bgc, dtype=int)
    for i, j in arrow:
        gi[h - 3 + i, w // 2 - 1 + j] = indc
    inds = {(i, j) for i in range(objh) for j in range(objw)}
    def nbrs(s):
        out = set()
        for i, j in s:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        out.add((i + di, j + dj))
        return out
    sp = random.choice(sorted(inds))
    objx = {sp}
    numcd = _unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = random.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for _ in range(numc - 1):
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    while max(j for _, j in objx) - min(j for _, j in objx) + 1 != objw:
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    mi = min(i for i, _ in objx); mj = min(j for _, j in objx)
    objx = {(i - mi, j - mj) for i, j in objx}
    oh = max(i for i, _ in objx) + 1
    ow = max(j for _, j in objx) + 1
    loci = random.randint(0, h - 3 - oh)
    locj = w // 2 - ow // 2
    plcd = {(i + loci, j + locj) for i, j in objx}
    for i, j in plcd:
        gi[i, j] = objc
    go = gi.copy()
    lo, hi_ = locj, locj + ow - 1
    for i, j in plcd:
        jj = lo + hi_ - j + ow * sgn
        if 0 <= jj < w:                      # the original fill() clips off-grid cells
            go[i, jj] = objc
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def _edge_dirs(cells):
    """Directions (dr, dc) of the 3x3-box sides that hold two of the arrow's cells."""
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    out = []
    if sum(1 for r in rs if r == max(rs)) == 2: out.append((1, 0))
    if sum(1 for r in rs if r == min(rs)) == 2: out.append((-1, 0))
    if sum(1 for c in cs if c == max(cs)) == 2: out.append((0, 1))
    if sum(1 for c in cs if c == min(cs)) == 2: out.append((0, -1))
    return out


def _arrow_like(cells):
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    if len(cells) != 5 or max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
        return False
    r0, c0 = min(rs), min(cs)
    corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
    border = {(r0 + i, c0 + j) for i in range(3) for j in range(3)} - {(r0 + 1, c0 + 1)}
    return len(cells & corners) == 1 and len(cells & border) == 4 and len(_edge_dirs(cells)) == 1


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the colour all demonstration inputs are drawn on
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
    if len(groups) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the duplicated object's colour is the one the demonstrations grow
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int); ex_o = np.asarray(ex_o, dtype=int)
        if ex_i.shape != ex_o.shape:
            continue
        for v in np.unique(ex_o).tolist():
            if int((ex_o == v).sum()) > int((ex_i == v).sum()):
                grown[int(v)] += 1
    main_col = next((v for v, _ in grown.most_common() if v in groups), None)
    if main_col is None:
        arrows = [k for k in groups if _arrow_like(groups[k])]
        rest = [k for k in groups if k not in arrows]
        main_col = max(rest or list(groups), key=lambda k: len(groups[k]))
    arrow_col = next(k for k in groups if k != main_col)
    main, arrow = groups[main_col], groups[arrow_col]

    # the arrow points at the side of its 3x3 box that carries two cells
    d = (_edge_dirs(arrow) or [(1, 0)])[0]

    r0 = min(r for r, _ in main); r1 = max(r for r, _ in main)
    c0 = min(c for _, c in main); c1 = max(c for _, c in main)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    vertical = d[1] == 0

    # The unit is the object's bounding strip that touches the edge the arrow
    # points at; it is reflected across that edge.  If the grid border is closer
    # than the object's thickness, only the strip that fits beyond the edge is
    # reflected (the rest would land off-grid).
    if d == (1, 0):
        k = min(oh, hi - 1 - r1); src = (r1 - k + 1, c0, k, ow); dst = (r1 + 1, c0)
    elif d == (-1, 0):
        k = min(oh, r0); src = (r0, c0, k, ow); dst = (r0 - k, c0)
    elif d == (0, 1):
        k = min(ow, wi - 1 - c1); src = (r0, c1 - k + 1, oh, k); dst = (r0, c1 + 1)
    else:
        k = min(ow, c0); src = (r0, c0, oh, k); dst = (r0, c0 - k)
    sr, sc, sh, sw = src
    dr_, dc_ = dst

    if k > 0 and main_col != 0:
        # copy the strip (background included) from the input, lay it beside
        # itself on the arrow's side, then mirror the laid-down rectangle in place.
        ops.append(28); sels.append([sr, sc, sh - 1, sw - 1])       # CopyI full rectangle
        ops.append(30); sels.append([dr_, dc_, 0, 0])               # Paste beside it
        strip = I[sr:sr + sh, sc:sc + sw]
        flipped = strip[::-1, :] if vertical else strip[:, ::-1]
        if not np.array_equal(strip, flipped):                      # symmetric -> flip is a no-op
            ops.append(27 if vertical else 26)
            sels.append([dr_, dc_, sh - 1, sw - 1])                 # whole pasted rectangle
    elif k > 0:
        # colour 0 is transparent to Copy/Paste: draw the reflected strip directly
        cells = []
        for r, c in main:
            if not (sr <= r < sr + sh and sc <= c < sc + sw):
                continue
            if vertical:
                cells.append((dr_ + (sr + sh - 1 - r), c))
            else:
                cells.append((r, dc_ + (sc + sw - 1 - c)))
        ops.append(0); sels.append(sel_of(sorted(cells)))

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
