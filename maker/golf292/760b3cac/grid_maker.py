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
from maker.sel_helpers import sel_of
from collections import Counter
import numpy as np
import random


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # object colour kept non-zero: 0 is transparent to Copy/Paste
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    indc = random.choice([c for c in cols if c not in (bgc, objc)])
    return {"bgc": bgc, "objc": objc, "indc": indc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, indc) -> dict:
    objL = {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)}
    objL = frozenset(objL)
    objR = vmirror(objL)
    h = unifint(diff_lb, diff_ub, (5, min(30, max_h)))
    kmax = max(3, min(14, (min(30, max_w) - 1) // 2))
    w = unifint(diff_lb, diff_ub, (3, kmax))
    w = 2 * w + 1
    rotf = choice((identity, rot90, rot180, rot270))
    if rotf in (rot90, rot270) and (w > max_h or h > max_w):
        rotf = identity
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    if choice((True, False)):
        obj = objL
        sgn = -1
    else:
        obj = objR
        sgn = 1
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))
    inds = asindices(c)
    sp = choice(totuple(inds))
    objx = {sp}
    numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for k in range(numc - 1):
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    while width(objx) != objw:
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    objx = normalize(frozenset(objx))
    objh, objw = shape(objx)
    loci = randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)
    objx2 = vmirror(plcd)
    plcd2 = shift(objx2, (0, objw * sgn))
    go = fill(gi, objc, plcd2)
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _arrow_dir(cells):
    # the side of the arrow's 3x3 box holding two of its cells
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    if sum(1 for r, _ in cells if r == r1) == 2:
        return (1, 0)
    if sum(1 for r, _ in cells if r == r0) == 2:
        return (-1, 0)
    if sum(1 for _, c in cells if c == c1) == 2:
        return (0, 1)
    if sum(1 for _, c in cells if c == c0) == 2:
        return (0, -1)
    return None


def _is_arrow(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if len(cells) != 5 or max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
        return False
    r0, c0 = min(rs), min(cs)
    corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
    return len(set(cells) & corners) == 1 and (r0 + 1, c0 + 1) in cells


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the colour every input (this one and the demonstrations) is drawn on
    cnt = Counter(I.flatten().tolist())
    for ex_i, _ in (examples or []):
        cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    groups = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                groups.setdefault(int(I[r, c]), []).append((r, c))
    if len(groups) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # the object colour: the one the demonstrations grow
    main_col = None
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int)
        ex_o = np.asarray(ex_o, dtype=int)
        if ex_i.shape == ex_o.shape:
            for v in np.unique(ex_o).tolist():
                if (ex_o == v).sum() > (ex_i == v).sum():
                    grown[int(v)] += 1
    for v, _ in grown.most_common():
        if v in groups:
            main_col = v
            break
    if main_col is None:
        # no demonstrations: the marker is the arrow lying against a canvas edge,
        # pointing along that edge
        def marker_like(cells):
            if not _is_arrow(cells):
                return False
            d0 = _arrow_dir(cells)
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            on_tb = min(rs) == 0 or max(rs) == hi - 1
            on_lr = min(cs) == 0 or max(cs) == wi - 1
            return (on_tb and d0[0] == 0) or (on_lr and d0[1] == 0)
        arrows = [k for k in groups if marker_like(groups[k])]
        rest = [k for k in groups if k not in arrows] or list(groups)
        main_col = max(rest, key=lambda k: len(groups[k]))
    arrow_col = [k for k in groups if k != main_col][0]
    d = _arrow_dir(groups[arrow_col]) or (1, 0)

    main = groups[main_col]
    rs = [r for r, _ in main]
    cs = [c for _, c in main]
    sr, sc = min(rs), min(cs)
    oh, ow = max(rs) - sr + 1, max(cs) - sc + 1

    # mirrored copy lands next to the object, one object-size along d.
    # keep only the part of the source whose mirror lands on the canvas.
    if d[1] == 0:
        flip_op = 27
        dr, dc = sr + d[0] * oh, sc
        lo, hi_ = max(0, dr), min(hi, dr + oh)          # on-grid dst rows
        if hi_ <= lo:
            return [34], [[0, 0, hi - 1, wi - 1]]
        # dst row dr+j  <-  src row sr+oh-1-j
        src_r0 = sr + oh - 1 - (hi_ - 1 - dr)
        src = [src_r0, sc, hi_ - lo - 1, ow - 1]
        dst = [lo, dc, hi_ - lo - 1, ow - 1]
    else:
        flip_op = 26
        dr, dc = sr, sc + d[1] * ow
        lo, hi_ = max(0, dc), min(wi, dc + ow)
        if hi_ <= lo:
            return [34], [[0, 0, hi - 1, wi - 1]]
        src_c0 = sc + ow - 1 - (hi_ - 1 - dc)
        src = [sr, src_c0, oh - 1, hi_ - lo - 1]
        dst = [dr, lo, oh - 1, hi_ - lo - 1]

    # copy the object's full rectangle (background included, on purpose),
    # paste it beside the object, mirror that full rectangle in place
    ops.append(28); sels.append(src)                     # CopyI (full rectangle)
    ops.append(30); sels.append([dst[0], dst[1], 0, 0])  # Paste
    block = I[src[0]:src[0] + src[2] + 1, src[1]:src[1] + src[3] + 1]
    flipped = np.flipud(block) if flip_op == 27 else np.fliplr(block)
    if not np.array_equal(block, flipped):               # a symmetric block needs no flip
        ops.append(flip_op); sels.append(dst)            # Flip (full rectangle)

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
