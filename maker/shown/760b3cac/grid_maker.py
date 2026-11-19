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
def sample_colors(num_examples=None) -> dict:
    import random
    # (rot_index, sgn) combinations that yield each final direction in the rotated frame.
    # unrotated displacement is (0, sgn); rot90(CW): (dr,dc)->(dc,-dr); rot180: ->(-dr,-dc);
    # rot270(CCW): (dr,dc)->(-dc,dr)
    DIR_VARIANTS = {
        0: [(0, 1), (2, -1)],    # copy goes RIGHT
        1: [(1, 1), (3, -1)],    # copy goes DOWN
        2: [(2, 1), (0, -1)],    # copy goes LEFT
        3: [(3, 1), (1, -1)],    # copy goes UP
    }
    cols = list(range(10))
    bgc, objc, indc = random.sample(cols, 3)

    n_ex = num_examples if num_examples else 3
    dirs = [0, 1, 2, 3]
    if n_ex >= len(dirs):
        chosen = list(dirs)
        chosen += [random.choice(dirs) for _ in range(n_ex - len(dirs))]
        random.shuffle(chosen)
    else:
        chosen = random.sample(dirs, n_ex)

    examples = []
    for d in chosen:
        rot, sgn = random.choice(DIR_VARIANTS[d])
        examples.append({"rot": rot, "sgn": sgn})
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "objc": objc, "indc": indc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, objc=None, indc=None, rot=None, sgn=None) -> dict:
    import random
    if rot is None:
        rot = random.choice((0, 1, 2, 3))
    if sgn is None:
        sgn = random.choice((-1, 1))

    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)

    # the final rotation swaps the canvas dimensions for odd k
    if rot % 2 == 0:
        hcap, wcap = min(max_h, 30), min(max_w, 30)
    else:
        hcap, wcap = min(max_w, 30), min(max_h, 30)
    hcap = max(5, hcap)
    wcap = max(7, wcap)

    h = unifint(diff_lb, diff_ub, (5, hcap))
    wub = min(14, max(3, (wcap - 1) // 2))
    ww = unifint(diff_lb, diff_ub, (3, wub))
    w = 2 * ww + 1

    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objwp = unifint(diff_lb, diff_ub, (1, max(1, w // 6)))
    objw = 2 * objwp + 1

    # build the payload blob; reject blobs that look like the 3x3 / 5-cell arrow marker
    while True:
        c = canvas(-1, (objh, objw))
        inds = asindices(c)
        sp = random.choice(totuple(inds))
        objx = {sp}
        numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
        numc = random.choice((numcd, objh * objw - numcd))
        numc = min(max(1, numc), objh * objw)
        for k in range(numc - 1):
            objx.add(random.choice(totuple((inds - objx) & mapply(neighbors, objx))))
        while width(objx) != objw:
            objx.add(random.choice(totuple((inds - objx) & mapply(neighbors, objx))))
        objx = normalize(objx)
        if not (shape(objx) == (3, 3) and len(objx) == 5):
            break

    oh2, ow2 = shape(objx)

    gi = canvas(bgc, (h, w))
    obj = objL if sgn == -1 else objR
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))

    loci = random.randint(0, h - 3 - oh2)
    locj = w // 2 - ow2 // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)

    plcd2 = shift(vmirror(plcd), (0, ow2 * sgn))
    go = fill(gi, objc, plcd2)

    rotf = (identity, rot90, rot180, rot270)[rot]
    return {'input': rotf(gi), 'output': rotf(go)}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for (r, c) in cells]}

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- background: shared by every instance of the episode -------------
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            g = np.asarray(ex[0], dtype=int)
            cnt.update(g.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    def fg_groups(g):
        d = {}
        for r in range(g.shape[0]):
            for cc in range(g.shape[1]):
                v = int(g[r, cc])
                if v != bgc:
                    d.setdefault(v, set()).add((r, cc))
        return d

    def bbox_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    def is_marker(cells):
        # the arrow marker: 3x3 bbox, 5 cells, 4 of them on the bbox border,
        # exactly 1 of them on a bbox corner
        if len(cells) != 5:
            return False
        r0, r1, c0, c1 = bbox_of(cells)
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            return False
        box = {(r, c) for r in (r0, r1) for c in range(c0, c1 + 1)}
        box |= {(r, c) for c in (c0, c1) for r in range(r0, r1 + 1)}
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        return len(cells & box) == 4 and len(cells & corners) == 1

    groups = fg_groups(I)

    # ---- which colour is the marker: read it from the demonstrations -----
    marker_color = None
    if examples:
        common = None
        for ex in examples:
            g = np.asarray(ex[0], dtype=int)
            d = {}
            for r in range(g.shape[0]):
                for cc in range(g.shape[1]):
                    v = int(g[r, cc])
                    if v != bgc:
                        d.setdefault(v, set()).add((r, cc))
            cand = {col for col, cells in d.items() if is_marker(cells)}
            common = cand if common is None else (common & cand)
        if common:
            usable = [col for col in common if col in groups and is_marker(groups[col])]
            if len(usable) >= 1:
                marker_color = sorted(usable)[0]
    if marker_color is None:
        cand = [col for col, cells in groups.items() if is_marker(cells)]
        marker_color = sorted(cand)[0] if cand else sorted(groups)[0]

    ind = groups[marker_color]
    others = [col for col in groups if col != marker_color]
    objc = others[0]
    payload = groups[objc]

    # ---- direction the marker points at ---------------------------------
    ir0, ir1, ic0, ic1 = bbox_of(ind)
    top = sum(1 for (r, c) in ind if r == ir0)
    bottom = sum(1 for (r, c) in ind if r == ir1)
    left = sum(1 for (r, c) in ind if c == ic0)
    right = sum(1 for (r, c) in ind if c == ic1)
    if bottom == 2:
        d = (1, 0)
    elif top == 2:
        d = (-1, 0)
    elif right == 2:
        d = (0, 1)
    else:
        d = (0, -1)

    # ---- payload bbox and the destination one full extent away ----------
    r0, r1, c0, c1 = bbox_of(payload)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1
    dr = d[0] * oh
    dc = d[1] * ow
    tr, tc = r0 + dr, c0 + dc

    vertical = (d[0] != 0)
    if vertical:
        mirrored = {(r0 + r1 - r, c) for (r, c) in payload}
        flip_op = 27                      # FlipV = flipud
    else:
        mirrored = {(r, c0 + c1 - c) for (r, c) in payload}
        flip_op = 26                      # FlipH = fliplr
    symmetric = (mirrored == payload)
    dest_cells = sorted((r + dr, c + dc) for (r, c) in mirrored)

    ops, sels = [], []

    if objc != 0:
        # duplicate the payload's whole bbox rectangle (background included),
        # then mirror that rectangle in place -> bbox selections are exact here
        ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])        # CopyI source rect
        ops.append(30); sels.append([tr, tc, 0, 0])                  # Paste at destination
        if not symmetric:
            # in-place mirror of the freshly pasted rectangle (full rectangle on purpose)
            ops.append(flip_op); sels.append([tr, tc, oh - 1, ow - 1])
    else:
        # payload colour is 0: Copy/Paste treats it as "nothing", so draw the
        # mirrored copy directly on the (all-background) destination region
        ops.append(0); sels.append(sel_of(dest_cells))

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
