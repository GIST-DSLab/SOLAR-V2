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


def sample_colors(num_examples=None) -> dict:
    # The rule only works if the solver can tell the duplicated object from
    # the arrow marker, so all three colours stay fixed for the whole episode.
    bgc, objc, indc = random.sample(range(10), 3)
    return {"bgc": bgc, "objc": objc, "indc": indc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, indc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, round(random.uniform(lo, hi))))

    if bgc is None or objc is None or indc is None:
        bgc, objc, indc = random.sample(range(10), 3)

    # A random rotation can swap h and w, so both must fit inside both limits.
    m = min(max_h, max_w)
    objL = {(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)}
    objR = {(r, 2 - c) for (r, c) in objL}
    h = unifint(diff_lb, diff_ub, (5, max(5, m)))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(14, (m - 1) // 2))))
    w = 2 * w + 1
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, max(1, w // 6)))
    objw = 2 * objw + 1
    gi = [[bgc] * w for _ in range(h)]
    if random.choice((True, False)):
        obj, sgn = objL, -1
    else:
        obj, sgn = objR, 1
    for (r, c) in obj:
        gi[r + h - 3][c + w // 2 - 1] = indc
    inds = {(i, j) for i in range(objh) for j in range(objw)}
    sp = random.choice(sorted(inds))
    objx = {sp}

    def nbrs(cs):
        out = set()
        for (a, b) in cs:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        out.add((a + da, b + db))
        return out

    numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = random.choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for _ in range(numc - 1):
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    while len({c for _, c in objx}) != objw:
        objx.add(random.choice(sorted((inds - objx) & nbrs(objx))))
    mr = min(r for r, _ in objx); mc = min(c for _, c in objx)
    objx = {(r - mr, c - mc) for r, c in objx}
    objh = max(r for r, _ in objx) + 1
    objw = max(c for _, c in objx) + 1
    loci = random.randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = {(r + loci, c + locj) for r, c in objx}
    for (r, c) in plcd:
        gi[r][c] = objc
    go = [row[:] for row in gi]
    for (r, c) in plcd:
        mc2 = locj + (locj + objw - 1 - c) + objw * sgn
        go[r][mc2] = objc
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(np.array(gi), -k)  # rot90 (CW) applied k times
    go = np.rot90(np.array(go), -k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading shared with the code-golf entry and the maker: the object's whole
    # bounding rectangle is one unit. It is reproduced mirrored in the slot right
    # next to it, on the side the arrow's two-cell edge points to. The readings
    # agree, so the route is the same: CopyI the rectangle, Paste it into the
    # adjacent slot, then mirror that slot in place.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

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

    def is_arrow(cells):
        rs = [r for r, _ in cells]; cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        if len(cells) != 5 or max(rs) - r0 != 2 or max(cs) - c0 != 2:
            return False
        corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
        return len(cells & corners) == 1

    # The duplicated object's colour is the one the demonstrations make grow.
    main_col = None
    grown = Counter()
    for ex_i, ex_o in (examples or []):
        ex_i = np.asarray(ex_i, dtype=int); ex_o = np.asarray(ex_o, dtype=int)
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
        arrows = [k for k in groups if is_arrow(groups[k])]
        rest = [k for k in groups if k not in arrows]
        main_col = max(rest or list(groups), key=lambda k: len(groups[k]))
    others = [k for k in groups if k != main_col]
    main = groups[main_col]
    arrow = groups[others[0]] if others else set()

    # Direction: the side of the arrow's 3x3 box that holds two of its cells.
    dr, dc = 1, 0
    if arrow:
        rs = [r for r, _ in arrow]; cs = [c for _, c in arrow]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bot = sum(1 for r, _ in arrow if r == r1)
        top = sum(1 for r, _ in arrow if r == r0)
        rgt = sum(1 for _, c in arrow if c == c1)
        lft = sum(1 for _, c in arrow if c == c0)
        dr = (bot == 2) - (top == 2)
        dc = (rgt == 2) - (lft == 2)
        if dr == 0 and dc == 0:
            dr = 1

    src_r = min(r for r, _ in main); src_c = min(c for _, c in main)
    oh = max(r for r, _ in main) - src_r + 1
    ow = max(c for _, c in main) - src_c + 1
    dst_r, dst_c = src_r + dr * oh, src_c + dc * ow
    if dr != 0:
        mirrored = {(src_r + (oh - 1 - (r - src_r)) + dr * oh, c) for (r, c) in main}
    else:
        mirrored = {(r, src_c + (ow - 1 - (c - src_c)) + dc * ow) for (r, c) in main}
    pasted = {(r + dr * oh, c + dc * ow) for (r, c) in main}
    fits = 0 <= dst_r and dst_r + oh <= hi and 0 <= dst_c and dst_c + ow <= wi
    on_grid = sorted((r, c) for (r, c) in mirrored if 0 <= r < hi and 0 <= c < wi)

    if main_col != 0 and fits:
        # Full-rectangle selections on purpose: the object's whole bounding box
        # (background included) is the unit that gets copied and mirrored.
        ops.append(28); sels.append([src_r, src_c, oh - 1, ow - 1])   # CopyI
        ops.append(30); sels.append([dst_r, dst_c, 0, 0])             # Paste
        if pasted != mirrored:   # a self-symmetric object needs no flip
            ops.append(27 if dr != 0 else 26)                          # FlipV / FlipH
            sels.append([dst_r, dst_c, oh - 1, ow - 1])
    elif on_grid:
        # Copy/Paste skips colour 0, so draw the mirrored copy's cells directly.
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
