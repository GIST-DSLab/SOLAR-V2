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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    # objc must be non-zero: the object is relocated with CopyI/Paste, and 0 is
    # "nothing" to the clipboard.
    objc = random.choice([c for c in range(1, 10) if c != bgc])
    boxc = random.choice([c for c in cols if c not in (bgc, objc)])

    variants = [{"transposed": False}, {"transposed": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, boxc, transposed=None) -> dict:
    from random import randint, choice, sample
    if transposed is None:
        transposed = choice((True, False))
    lim_h, lim_w = (max_w, max_h) if transposed else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (10, lim_h))
    w = unifint(diff_lb, diff_ub, (4, lim_w))
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
    ncands = objh * (fullw - 2)
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, ncands // 2))
        ncells = choice((ncellsd, ncands - ncellsd))
        ncells = min(max(1, ncells), ncands)
        # anchor the object on the far edge of its band: this makes the reflected
        # copy land exactly against the bracket (keeps generator == verifier).
        anchor = (0, randint(1, fullw - 2))
        rest = [c for c in totuple(cands) if c != anchor]
        cells = frozenset([anchor] + sample(rest, ncells - 1))
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transposed:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I): a two-bracket frame sits in the middle; one blob of object
    cells lies outside each bracket.  Each blob is mirrored through its bracket and
    tucked inside the frame, flush against the bracket's end row/col; its old place
    becomes background.

    Per blob: CopyI its bbox -> Paste inside the frame -> flip it in place
    (only if the flip actually changes it) -> erase the blob's old bbox.
    """
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    pts_of = {}
    for c in np.unique(I):
        pts_of[int(c)] = [(int(r), int(cc)) for r, cc in np.argwhere(I == c)]

    def bbox(pts):
        rs = [p[0] for p in pts]
        cs = [p[1] for p in pts]
        return min(rs), max(rs), min(cs), max(cs)

    def is_frame(pts):
        r0, r1, c0, c1 = bbox(pts)
        for (r, c) in pts:
            if r0 < r < r1 and c0 < c < c1:
                return False
        H = r1 - r0 + 1
        W = c1 - c0 + 1
        rowcnt = Counter(r for r, _ in pts)
        colcnt = Counter(c for _, c in pts)
        if len(pts) == 2 * W + 4 and rowcnt[r0] == W and rowcnt[r1] == W:
            return True
        if len(pts) == 2 * H + 4 and colcnt[c0] == H and colcnt[c1] == H:
            return True
        return False

    boxc = None
    for c in sorted(pts_of):
        if len(pts_of[c]) >= 6 and is_frame(pts_of[c]):
            boxc = c
            break
    rest = [c for c in pts_of if c != boxc]
    objc = min(rest, key=lambda c: len(pts_of[c]))
    bgc = [c for c in rest if c != objc][0]

    box = pts_of[boxc]
    br0, br1, bc0, bc1 = bbox(box)
    # brackets are the two full lines: horizontal if the frame occupies its own
    # centre column (a full-width line does; two vertical bars do not).
    horizontal = any(c == bc0 + (bc1 - bc0 + 1) // 2 for _, c in box)

    obj = pts_of[objc]
    if horizontal:
        groups = [([p for p in obj if p[0] < br0], 'near'),
                  ([p for p in obj if p[0] > br1], 'far')]
    else:
        groups = [([p for p in obj if p[1] < bc0], 'near'),
                  ([p for p in obj if p[1] > bc1], 'far')]

    for grp, side in groups:
        if not grp:
            continue
        u, l, a, b = bbox(grp)
        hh = l - u + 1
        ww = b - a + 1
        rel = {(r - u, c - a) for r, c in grp}
        if horizontal:
            dr = br0 + 2 if side == 'near' else br1 - 2 - (hh - 1)
            dc = a
            flip_op = 27                                  # up<->down
            same = rel == {(hh - 1 - r, c) for r, c in rel}
        else:
            dr = u
            dc = bc0 + 2 if side == 'near' else bc1 - 2 - (ww - 1)
            flip_op = 26                                  # left<->right
            same = rel == {(r, ww - 1 - c) for r, c in rel}

        ops.append(28); sels.append([u, a, hh - 1, ww - 1])      # grab the blob
        ops.append(30); sels.append([dr, dc, 0, 0])              # drop it inside the frame
        if not same:
            ops.append(flip_op); sels.append([dr, dc, hh - 1, ww - 1])
        ops.append(bgc); sels.append([u, a, hh - 1, ww - 1])     # clear its old place

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
