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


# ---------------------------------------------------------------- variants
# Discrete structure = which way the 5-cell arrow points (= which side the
# mirrored copy goes).  rot_idx picks the whole-grid rotation, sgn picks the
# arrow's handedness.  (rot_idx 0/1) x (sgn +-1) already covers all four
# resulting directions (right, left, and their rotations down/up).
CORE_VARIANTS = [
    {"rot_idx": 0, "sgn": 1},
    {"rot_idx": 0, "sgn": -1},
    {"rot_idx": 1, "sgn": 1},
    {"rot_idx": 1, "sgn": -1},
]
EXTRA_VARIANTS = [
    {"rot_idx": 2, "sgn": 1},
    {"rot_idx": 2, "sgn": -1},
    {"rot_idx": 3, "sgn": 1},
    {"rot_idx": 3, "sgn": -1},
]
VARIANTS = CORE_VARIANTS + EXTRA_VARIANTS


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # objc must be non-zero: the mirrored copy is produced with CopyI/Paste,
    # and 0 is "transparent" for those ops.
    objc = random.choice([c for c in cols if c != bgc and c != 0])
    indc = random.choice([c for c in cols if c not in (bgc, objc)])

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
    """3x3 bbox, 5 cells, 4 of them on the box perimeter, exactly 1 on a corner."""
    cells = set(cells)
    if len(cells) != 5:
        return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    if max(rs) - r0 != 2 or max(cs) - c0 != 2:
        return False
    if (r0 + 1, c0 + 1) not in cells:          # centre present -> 4 on the box
        return False
    corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
    return len(cells & corners) == 1


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, indc,
             rot_idx=None, sgn=None) -> dict:
    if rot_idx is None or sgn is None:
        v = random.choice(VARIANTS)
        rot_idx, sgn = v["rot_idx"], v["sgn"]

    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)

    # a 90/270 rotation swaps the final dimensions
    if rot_idx % 2 == 1:
        h_lim, w_lim = max_w, max_h
    else:
        h_lim, w_lim = max_h, max_w

    h = unifint(diff_lb, diff_ub, (5, max(5, h_lim)))
    wub = min(14, (w_lim - 1) // 2)
    wpre = unifint(diff_lb, diff_ub, (4, max(4, wub)))
    w = 2 * wpre + 1

    obj = objL if sgn == -1 else objR

    objh = unifint(diff_lb, diff_ub, (1, max(1, h - 3)))
    # widest odd objw whose mirrored copy still fits fully inside the canvas
    uub = max(1, min(w // 6, (w // 3 - 1) // 2))
    objw = 2 * unifint(diff_lb, diff_ub, (1, uub)) + 1

    gi = canvas(bgc, (h, w))
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))

    for _attempt in range(64):
        c = canvas(-1, (objh, objw))
        inds = asindices(c)
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
        while width(objx) != objw and guard < 500:
            cand = totuple((inds - objx) & mapply(neighbors, objx))
            if len(cand) == 0:
                break
            objx.add(choice(cand))
            guard += 1
        if width(objx) != objw:
            continue
        objx = normalize(objx)
        # the arrow must stay the unique arrow-shaped object
        if not _is_arrow_like(objx):
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


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- split the foreground into its two colour groups (as in I only) ----
    groups = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                groups.setdefault(v, set()).add((r, c))

    arrow_col = None
    for col, cells in groups.items():
        if _is_arrow_like(cells):
            arrow_col = col
            break
    main_col = [c for c in groups if c != arrow_col][0]
    arrow = groups[arrow_col]
    main = groups[main_col]

    # ---- read the direction off the arrow: the side holding 2 of its cells --
    ar = [r for r, _ in arrow]
    ac = [c for _, c in arrow]
    at, ab, al, arr = min(ar), max(ar), min(ac), max(ac)
    n_top = sum(1 for r, _ in arrow if r == at)
    n_bot = sum(1 for r, _ in arrow if r == ab)
    n_left = sum(1 for _, c in arrow if c == al)
    n_right = sum(1 for _, c in arrow if c == arr)
    if n_bot == 2:
        dr, dc = 1, 0
    elif n_top == 2:
        dr, dc = -1, 0
    elif n_right == 2:
        dr, dc = 0, 1
    else:
        dr, dc = 0, -1

    # ---- the copy is the main object mirrored across the axis perpendicular
    #      to that direction, laid down exactly one object-size along it ------
    mr = [r for r, _ in main]
    mc = [c for _, c in main]
    r0, c0 = min(mr), min(mc)
    oh = max(mr) - r0 + 1
    ow = max(mc) - c0 + 1
    dst_r = r0 + oh * dr
    dst_c = c0 + ow * dc
    flip_op = 27 if dc == 0 else 26   # vertical move -> up/down mirror

    ops, sels = [], []
    # bbox selections below are intentionally the FULL rectangles: we duplicate
    # the whole object region (background included) and mirror that region.
    ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])        # CopyI source
    ops.append(30); sels.append([dst_r, dst_c, 0, 0])            # Paste copy
    ops.append(flip_op); sels.append([dst_r, dst_c, oh - 1, ow - 1])  # mirror it

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
