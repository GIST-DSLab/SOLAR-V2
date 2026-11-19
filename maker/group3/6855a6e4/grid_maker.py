"""
ARC Task: 6855a6e4 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # the pattern colour travels with ARCLE object ops, which never write 0
    objc = random.choice([c for c in cols if c not in (bgc, 0)])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    HH, WW = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, HH))
    w = unifint(diff_lb, diff_ub, (4, WW))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        # the pattern always reaches the far row of its band, so "mirror the
        # pattern and lay it flush against the inner side of the rail" is exact
        for _ in range(100):
            cells = frozenset(sample(totuple(cands), ncells))
            if min(i for i, j in cells) == 0:
                break
        else:
            raise ValueError("no far-row pattern")
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    flat = [v for row in gi for v in row]
    if flat.count(objc) >= flat.count(bgc):
        raise ValueError("pattern colour must be the minority colour")
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []

    # --- the frame: a colour whose cells all lie on its own bbox border and
    #     number 2*side+4 (two full rails plus four inward stubs) ---
    cnt = Counter(I.flatten().tolist())
    boxc, horizontal = None, True
    for c in sorted(cnt):
        pts = np.argwhere(I == c)
        r0, r1 = pts[:, 0].min(), pts[:, 0].max()
        c0, c1 = pts[:, 1].min(), pts[:, 1].max()
        if not all(r in (r0, r1) or cc in (c0, c1) for r, cc in pts):
            continue
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        rails_h = np.all(I[r0, c0:c1 + 1] == c) and np.all(I[r1, c0:c1 + 1] == c)
        rails_v = np.all(I[r0:r1 + 1, c0] == c) and np.all(I[r0:r1 + 1, c1] == c)
        if rails_h and len(pts) == 2 * ww + 4:
            boxc, horizontal = c, True
            break
        if rails_v and len(pts) == 2 * hh + 4:
            boxc, horizontal = c, False
            break
    rest = [c for c in cnt if c != boxc]
    objc = min(rest, key=lambda c: cnt[c])      # the pattern is the rarer colour
    bgc = [c for c in rest if c != objc][0]

    bx = np.argwhere(I == boxc)
    br0, br1 = int(bx[:, 0].min()), int(bx[:, 0].max())
    bc0, bc1 = int(bx[:, 1].min()), int(bx[:, 1].max())

    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    # axis perpendicular to the rails
    ax = 0 if horizontal else 1
    lo_rail, hi_rail = (br0, br1) if horizontal else (bc0, bc1)
    near = [p for p in obj if p[ax] < lo_rail]      # pattern outside the low rail
    far = [p for p in obj if p[ax] > hi_rail]       # pattern outside the high rail

    flip_op = 27 if horizontal else 26              # mirror across the rail direction
    for cells, side in ((near, 0), (far, 1)):
        if not cells:
            continue
        a = min(p[ax] for p in cells)
        b = max(p[ax] for p in cells)
        # destination: the mirrored pattern sits flush with the inner side of
        # its own rail (just past the stub row)
        if side == 0:
            shift_n = (lo_rail + 2) - a
            mv = 21 if horizontal else 22           # down / right
        else:
            shift_n = (hi_rail - 2) - b
            mv = 20 if horizontal else 23           # up / left
        steps = abs(shift_n)

        # 1. mirror the pattern object in place (this grabs it as an object)
        sub = np.zeros((H, W), dtype=bool)
        for r, c in cells:
            sub[r, c] = True
        rr0, rr1 = min(r for r, _ in cells), max(r for r, _ in cells)
        cc0, cc1 = min(c for _, c in cells), max(c for _, c in cells)
        box = sub[rr0:rr1 + 1, cc0:cc1 + 1]
        mirrored = np.flipud(box) if horizontal else np.fliplr(box)
        grabbed = False
        if not np.array_equal(box, mirrored):
            ops.append(flip_op); sels.append(sel_of(cells))
            grabbed = True
        cur = [(rr0 + i, cc0 + j) for i, j in zip(*np.nonzero(mirrored))]

        # 2. slide the (mirrored) pattern across its rail into the frame:
        #    one grab, then empty selections continue the same object
        d = (1 if shift_n > 0 else -1)
        for k in range(steps):
            if not grabbed:
                ops.append(mv); sels.append(sel_of(cur))
                grabbed = True
            else:
                ops.append(mv); sels.append(sel_of([]))
            cur = [(r + d, c) if horizontal else (r, c + d) for r, c in cur]

        # 3. the grab zeroed the pattern's original footprint; restore background
        hole = sorted(set(cells) - set(cur))
        if bgc != 0 and hole:
            ops.append(int(bgc)); sels.append(sel_of(hole))

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6855a6e4"
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
                                f"for task 6855a6e4"
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
                    f"Failed to build a complete episode for task 6855a6e4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6855a6e4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
