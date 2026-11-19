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


def sample_colors(num_examples=None) -> dict:
    bgc, objc, boxc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "objc": objc, "boxc": boxc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, objc=None, boxc=None, transpose=None) -> dict:
    from dsl import (interval, canvas, connect, backdrop, fill, totuple, shift,
                     hmirror, paint, asobject, dmirror)
    from utils import unifint
    if bgc is None or objc is None or boxc is None:
        bgc, objc, boxc = random.sample(range(10), 3)
    if transpose is None:
        transpose = random.choice((True, False))
    hub = max(10, max_h if not transpose else max_w)
    wub = max(4, max_w if not transpose else max_h)
    h = unifint(diff_lb, diff_ub, (10, hub))
    w = unifint(diff_lb, diff_ub, (4, wub))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bcanv = canvas(bgc, (h, w))
    loci = random.randint(0, h - fullh)
    locj = random.randint(0, w - fullw)
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
        ncells = random.choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(random.sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if transpose:
        gi = dmirror(gi)
        go = dmirror(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    import numpy as np
    from collections import Counter
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [[int(r), int(c)] for r, c in cells]}

    I = np.asarray(I, dtype=int)
    H, W = I.shape
    G = I.copy()
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = int(cnt.most_common(1)[0][0])
    others = sorted(int(c) for c in cnt if c != bgc)

    # --- frame detection: two full rails + 4 inward stubs ---
    boxc, horizontal = None, True
    for c in others:
        pts = np.argwhere(I == c)
        r0, r1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        c0, c1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if not all((int(r) in (r0, r1)) or (int(cc) in (c0, c1)) for r, cc in pts):
            continue
        rows_full = all(I[r0, j] == c for j in range(c0, c1 + 1)) and \
            all(I[r1, j] == c for j in range(c0, c1 + 1))
        cols_full = all(I[i, c0] == c for i in range(r0, r1 + 1)) and \
            all(I[i, c1] == c for i in range(r0, r1 + 1))
        stubs_h = hh >= 4 and ww >= 3 and all(
            I[a, b] == c for a, b in ((r0 + 1, c0), (r0 + 1, c1), (r1 - 1, c0), (r1 - 1, c1)))
        stubs_v = ww >= 4 and hh >= 3 and all(
            I[a, b] == c for a, b in ((r0, c0 + 1), (r1, c0 + 1), (r0, c1 - 1), (r1, c1 - 1)))
        if rows_full and stubs_h and len(pts) == 2 * ww + 4:
            boxc, horizontal = c, True
            break
        if cols_full and stubs_v and len(pts) == 2 * hh + 4:
            boxc, horizontal = c, False
            break
    if boxc is None:
        boxc = others[0]
        pts = np.argwhere(I == boxc)
        r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        horizontal = all(I[r0, j] == boxc for j in range(c0, c1 + 1))
    objc = [c for c in others if c != boxc][0]

    bx = np.argwhere(I == boxc)
    br0, br1 = int(bx[:, 0].min()), int(bx[:, 0].max())
    bc0, bc1 = int(bx[:, 1].min()), int(bx[:, 1].max())

    obj = [(int(r), int(c)) for r, c in np.argwhere(I == objc)]
    if horizontal:
        groups = [[p for p in obj if p[0] < br0], [p for p in obj if p[0] > br1]]
    else:
        groups = [[p for p in obj if p[1] < bc0], [p for p in obj if p[1] > bc1]]

    # 1. all pattern cells leave their places outside the frame (all cells of one colour)
    ops.append(int(bgc)); sels.append(sel_of(obj))
    for r, c in obj:
        G[r, c] = bgc

    for gi, cells in enumerate(groups):
        if not cells:
            continue
        r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        rect = I[r0:r1 + 1, c0:c1 + 1]
        if horizontal:
            dest_c = c0
            dest_r = br0 + 2 if gi == 0 else br1 - 2 - (hh - 1)
            flip_op = 27
        else:
            dest_r = r0
            dest_c = bc0 + 2 if gi == 0 else bc1 - 2 - (ww - 1)
            flip_op = 26

        # 2. copy the pattern's rectangle from the input, lay it inside the frame
        ops.append(28); sels.append([r0, c0, hh - 1, ww - 1])
        ops.append(30); sels.append([dest_r, dest_c, 0, 0])
        blk = G[dest_r:dest_r + hh, dest_c:dest_c + ww]
        G[dest_r:dest_r + hh, dest_c:dest_c + ww] = np.where(rect != 0, rect, blk)

        # 3. zero-valued pattern cells don't travel with paste: lay them in
        rep = [(dest_r + i, dest_c + j) for i in range(hh) for j in range(ww)
               if rect[i, j] == 0 and G[dest_r + i, dest_c + j] != 0]
        if rep:
            ops.append(0); sels.append(sel_of(rep))
            for r, c in rep:
                G[r, c] = 0

        # 4. mirror the laid-down rectangle in place (whole rectangle is intended)
        cur = G[dest_r:dest_r + hh, dest_c:dest_c + ww].copy()
        flipped = np.flipud(cur) if horizontal else np.fliplr(cur)
        if not np.array_equal(cur, flipped):
            ops.append(flip_op); sels.append([dest_r, dest_c, hh - 1, ww - 1])
            G[dest_r:dest_r + hh, dest_c:dest_c + ww] = flipped

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
