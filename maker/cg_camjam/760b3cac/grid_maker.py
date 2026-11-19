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
from collections import Counter

import numpy as np

from maker.sel_helpers import sel_of


# ── episode sampling ──────────────────────────────────────────────────────────

def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, indc = random.sample(cols, 3)
    return {"bgc": bgc, "objc": objc, "indc": indc}


# ── generator ─────────────────────────────────────────────────────────────────

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, indc=None) -> dict:
    from dsl import (interval, vmirror, canvas, fill, shift, asindices, totuple,
                     mapply, neighbors, width, normalize, shape, identity,
                     rot90, rot180, rot270)
    from utils import unifint

    cols = interval(0, 10, 1)
    if bgc is None or objc is None or indc is None:
        bgc, objc, indc = random.sample(cols, 3)

    # the grid may be rotated, so both sides must fit either bound
    lim = min(max_h, max_w)
    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)
    h = unifint(diff_lb, diff_ub, (5, max(5, lim)))
    w = unifint(diff_lb, diff_ub, (3, max(3, (lim - 1) // 2)))
    w = 2 * w + 1
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    if random.choice((True, False)):
        obj = objL
        sgn = -1
    else:
        obj = objR
        sgn = 1
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))
    inds = asindices(c)
    sp = random.choice(totuple(inds))
    objx = {sp}
    numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = random.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for _ in range(numc - 1):
        objx.add(random.choice(totuple((inds - objx) & mapply(neighbors, objx))))
    while width(objx) != objw:
        objx.add(random.choice(totuple((inds - objx) & mapply(neighbors, objx))))
    objx = normalize(objx)
    objh, objw = shape(objx)
    loci = random.randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)
    objx2 = vmirror(plcd)
    plcd2 = shift(objx2, (0, objw * sgn))
    go = fill(gi, objc, plcd2)
    rotf = random.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


# ── helpers for derive_operations ─────────────────────────────────────────────

def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _is_arrow_like(cells):
    if len(cells) != 5:
        return False
    _, _, h, w = _bbox(cells)
    return h == 3 and w == 3


def _edge_direction(arrow):
    """The side of the arrow's 3x3 box that holds two of its cells points
    to where the mirrored copy goes."""
    r0, c0, h, w = _bbox(arrow)
    r1, c1 = r0 + h - 1, c0 + w - 1
    out = []
    if sum(1 for r, _ in arrow if r == r1) == 2:
        out.append((1, 0))
    if sum(1 for r, _ in arrow if r == r0) == 2:
        out.append((-1, 0))
    if sum(1 for _, c in arrow if c == c1) == 2:
        out.append((0, 1))
    if sum(1 for _, c in arrow if c == c0) == 2:
        out.append((0, -1))
    return out


def _copy_cells(main, d):
    """Mirror image of the object, placed flush against it on side d."""
    r0, c0, h, w = _bbox(main)
    dst_r, dst_c = r0 + d[0] * h, c0 + d[1] * w
    cells = set()
    for (r, c) in main:
        lr, lc = r - r0, c - c0
        if d[1] != 0:           # horizontal neighbour -> left/right mirror
            lc = w - 1 - lc
        else:                   # vertical neighbour -> up/down mirror
            lr = h - 1 - lr
        cells.add((dst_r + lr, dst_c + lc))
    return cells, (dst_r, dst_c, h, w)


# ── derive_operations ─────────────────────────────────────────────────────────

def derive_operations(I, O, examples=None):
    # The golf reading ("the new half is the object's block read backwards,
    # placed against the object's edge") is the same as this one: the object's
    # rectangle is copied, laid next to itself on the side the arrow marks,
    # and mirrored. Nothing to change, so this follows the same route.
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

    # the duplicated object's colour is the one the demonstrations grow
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
        if rest:
            main_col = max(rest, key=lambda k: len(groups[k]))
        else:
            main_col = max(groups, key=lambda k: len(groups[k]))
    others = [k for k in groups if k != main_col]
    main = groups[main_col]
    arrow = groups[others[0]] if others else set()

    cands = (_edge_direction(arrow) if arrow else []) or [(1, 0)]
    d = cands[0]
    cells, box = _copy_cells(main, d)

    dst_r, dst_c, oh, ow = box
    on_grid = sorted((r, c) for (r, c) in cells if 0 <= r < hi and 0 <= c < wi)
    fits = 0 <= dst_r and dst_r + oh <= hi and 0 <= dst_c and dst_c + ow <= wi
    src_r, src_c = dst_r - d[0] * oh, dst_c - d[1] * ow

    if main_col != 0 and fits:
        # whole-rectangle selections on purpose (background included):
        # the object's full box is what gets copied and mirrored.
        ops.append(28); sels.append([src_r, src_c, oh - 1, ow - 1])   # CopyI
        ops.append(30); sels.append([dst_r, dst_c, 0, 0])             # Paste
        pasted = {(dst_r + (r - src_r), dst_c + (c - src_c)) for (r, c) in main}
        if pasted != set(cells):       # a self-mirroring shape needs no flip
            ops.append(27 if d[1] == 0 else 26)
            sels.append([dst_r, dst_c, oh - 1, ow - 1])
    elif on_grid:
        # 0 is transparent to Copy/Paste, and an off-canvas copy cannot be
        # pasted, so draw the mirrored object directly instead.
        ops.append(main_col); sels.append(sel_of(on_grid))

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
