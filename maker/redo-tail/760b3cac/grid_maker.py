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
# ----------------------------------------------------------------- variants
# The only discrete structure is WHICH WAY the 5-cell arrow points, i.e. on
# which side of the object its mirrored copy is laid down.  sgn picks the
# arrow's handedness (left/right before the whole-grid rotation), rot_idx the
# rotation.  These four entries already realise all four visible directions:
#   (sgn +1, rot 0) -> right   (sgn -1, rot 0) -> left
#   (sgn +1, rot 1) -> down    (sgn +1, rot 3) -> up
from maker.sel_helpers import sel_of
from collections import Counter
import numpy as np
import random

CORE_VARIANTS = [
    {"sgn": 1,  "rot_idx": 0},
    {"sgn": -1, "rot_idx": 0},
    {"sgn": 1,  "rot_idx": 1},
    {"sgn": 1,  "rot_idx": 3},
]
VARIANTS = CORE_VARIANTS + [
    {"sgn": -1, "rot_idx": 1},
    {"sgn": -1, "rot_idx": 3},
    {"sgn": 1,  "rot_idx": 2},
    {"sgn": -1, "rot_idx": 2},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, objc, indc = sample(cols, 3)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(CORE_VARIANTS):
        examples = [dict(v) for v in CORE_VARIANTS]
        examples += [dict(random.choice(VARIANTS))
                     for _ in range(n_ex - len(CORE_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(CORE_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "indc": indc, "instance_plan": plan}


def _is_arrow_like(cells):
    """3x3 bbox, 5 cells, centre present (4 on the box), exactly 1 on a corner —
    the marker's signature.  A second object with it would make the instance
    ambiguous, so the generator never draws one."""
    cells = set(cells)
    if len(cells) != 5:
        return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    if max(rs) - r0 != 2 or max(cs) - c0 != 2:
        return False
    if (r0 + 1, c0 + 1) not in cells:
        return False
    corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
    return len(cells & corners) == 1


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, indc,
             sgn=None, rot_idx=None) -> dict:
    if sgn is None or rot_idx is None:
        v = random.choice(VARIANTS)
        sgn, rot_idx = v["sgn"], v["rot_idx"]

    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)

    # a 90/270 rotation swaps the final grid dimensions
    if rot_idx % 2 == 1:
        h_lim, w_lim = max_w, max_h
    else:
        h_lim, w_lim = max_h, max_w
    h_lim, w_lim = min(30, h_lim), min(30, w_lim)

    h = unifint(diff_lb, diff_ub, (5, max(5, h_lim)))
    wub = max(3, min(14, (w_lim - 1) // 2))
    w = 2 * unifint(diff_lb, diff_ub, (3, wub)) + 1

    obj = objL if sgn == -1 else objR

    objh = unifint(diff_lb, diff_ub, (1, max(1, h - 3)))
    objw = 2 * unifint(diff_lb, diff_ub, (1, max(1, w // 6))) + 1

    gi = canvas(bgc, (h, w))
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))

    c = canvas(-1, (objh, objw))
    inds = asindices(c)
    for _attempt in range(64):
        sp = choice(totuple(inds))
        objx = {sp}
        numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
        numc = choice((numcd, objh * objw - numcd))
        numc = min(max(1, numc), objh * objw)
        for k in range(numc - 1):
            cand = totuple((inds - objx) & mapply(neighbors, objx))
            if len(cand) == 0:
                break
            objx.add(choice(cand))
        guard = 0
        while width(objx) != objw and guard < 1000:
            cand = totuple((inds - objx) & mapply(neighbors, objx))
            if len(cand) == 0:
                break
            objx.add(choice(cand))
            guard += 1
        objx = normalize(objx)
        # the arrow has to stay the only arrow-shaped object on the canvas
        if width(objx) == objw and not _is_arrow_like(objx):
            break
    else:
        objx = normalize(frozenset({(0, j) for j in range(objw)}))
    oh, ow = shape(objx)

    loci = randint(0, max(0, h - 3 - oh))
    locj = w // 2 - ow // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)

    plcd2 = shift(vmirror(plcd), (0, ow * sgn))
    go = fill(gi, objc, plcd2)

    rotf = (identity, rot90, rot180, rot270)[rot_idx]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------ derive helpers
def _edge_direction(cells):
    """The arrow points to the side of its 3x3 box that holds 2 of its cells."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    t, b, l, rr = min(rs), max(rs), min(cs), max(cs)
    hits = []
    if sum(1 for r, _ in cells if r == b) == 2:
        hits.append((1, 0))
    if sum(1 for r, _ in cells if r == t) == 2:
        hits.append((-1, 0))
    if sum(1 for _, c in cells if c == rr) == 2:
        hits.append((0, 1))
    if sum(1 for _, c in cells if c == l) == 2:
        hits.append((0, -1))
    return hits


def _copy_cells(cells, d):
    """Mirror `cells` inside their own bbox across the axis perpendicular to d,
    then lay the result down one object-size further along d."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    dr, dc = d
    out = set()
    for (r, c) in cells:
        if dr != 0:                       # vertical move -> mirror up/down
            out.add((r0 + r1 - r + dr * oh, c))
        else:                             # horizontal move -> mirror left/right
            out.add((r, c0 + c1 - c + dc * ow))
    return out, (r0 + dr * oh, c0 + dc * ow, oh, ow)


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    diff = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != O[r, c]]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, set()).add((r, c))

    # the object that gets duplicated is the one the new cells are made of
    main_col = int(O[diff[0][0], diff[0][1]]) if diff else None
    if main_col not in groups:
        if not groups:
            return [34], [[0, 0, ho - 1, wo - 1]]
        main_col = max(groups, key=lambda k: len(groups[k]))
    others = [k for k in groups if k != main_col]
    main = groups[main_col]
    arrow = groups[others[0]] if others else set()

    # direction: read it off the arrow; keep the reading that actually
    # reproduces the new cells when the arrow shape is ambiguous
    cands = (_edge_direction(arrow) if arrow else []) or \
        [(1, 0), (-1, 0), (0, 1), (0, -1)]
    d, cells, box = None, None, None
    for cand in cands:
        cc, bb = _copy_cells(main, cand)
        on = {(r, c) for (r, c) in cc if 0 <= r < hi and 0 <= c < wi}
        if not diff or on == set(diff):
            d, cells, box = cand, cc, bb
            break
    if d is None:
        d, cells, box = cands[0], *_copy_cells(main, cands[0])

    dst_r, dst_c, oh, ow = box
    on_grid = sorted((r, c) for (r, c) in cells if 0 <= r < hi and 0 <= c < wi)
    fits = 0 <= dst_r and dst_r + oh <= hi and 0 <= dst_c and dst_c + ow <= wi
    src_r, src_c = dst_r - d[0] * oh, dst_c - d[1] * ow

    if main_col != 0 and fits:
        # duplicate the object's whole region, then mirror that region in place.
        # both bbox selections are FULL rectangles on purpose (background
        # included) — that is exactly what is copied / mirrored.
        ops.append(28); sels.append([src_r, src_c, oh - 1, ow - 1])   # CopyI
        ops.append(30); sels.append([dst_r, dst_c, 0, 0])             # Paste
        mirrored = {(dst_r + (r - src_r), dst_c + (c - src_c)) for (r, c) in main}
        if mirrored != set(cells):     # a self-mirroring shape needs no flip
            ops.append(27 if d[1] == 0 else 26)
            sels.append([dst_r, dst_c, oh - 1, ow - 1])
    elif on_grid:
        # 0 is transparent to Copy/Paste, and a copy running off the canvas
        # cannot be pasted — draw the mirrored object directly instead.
        ops.append(main_col); sels.append(sel_of(on_grid))

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
