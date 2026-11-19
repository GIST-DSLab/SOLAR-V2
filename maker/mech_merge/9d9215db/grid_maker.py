"""
ARC Task: 9d9215db (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
ROTS = [0, 1, 2, 3]          # identity / rot90 / rot180 / rot270


def sample_colors(num_examples=None) -> dict:
    """Only the background is a real colour role here: the rule (mirror the seed
    quadrant into all four quadrants) is colour independent.  The one genuinely
    discrete structural dimension is which corner the seed sits in, i.e. the
    final rotation, so that is pre-planned per instance."""
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- generator ---
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, rot=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if rot is None:
        rot = choice((0, 1, 2, 3))

    # rot90 / rot270 transpose the canvas, so size limits swap for those cases
    if rot in (1, 3):
        hlim, wlim = (max_w - 1) // 2, (max_h - 1) // 2
    else:
        hlim, wlim = (max_h - 1) // 2, (max_w - 1) // 2
    hlim = max(5, min(14, hlim))
    wlim = max(5, min(14, wlim))

    h = unifint(diff_lb, diff_ub, (5, hlim))
    w = unifint(diff_lb, diff_ub, (5, wlim))
    h = h * 2 + 1
    w = w * 2 + 1

    remcols = remove(bgc, cols)
    ub = min(h, w) // 4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2 * k + 1, 2 * k + 1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)

    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = sample(linsbase, nlines)
    colors = [choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)

    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2)
                   for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))

    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------------- operations ---
def _base_derive_operations(I, O, examples=None):
    """
    Rule (read off I alone):
      * every mark lives at odd (row, col) inside ONE corner quadrant;
      * a mark pair {(2k+1, 2k+3), (2k+3, 2k+1)} (same colour) shoots a ray of
        odd-spaced cells inward from each of its two members;
      * the finished quadrant is then replicated into the other three quadrants
        by mirroring about the centre column and the centre row.

    Trajectory:  draw the rays  ->  CopyO the seed quadrant / half, Paste it on
    the far side, repair the cells the palette painted 0 (ARCLE treats 0 as
    "nothing there", so they never travel), FlipH / FlipV the pasted block.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ch, cw = (H - 1) // 2, (W - 1) // 2

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    g = I.copy()

    # ---- which corner holds the seed ----------------------------------------
    fg = [(r, c) for r in range(H) for c in range(W) if I[r, c] != bgc]
    frs = [r for r, _ in fg]
    fcs = [c for _, c in fg]
    top = (max(frs) < ch) if fg else True
    left = (max(fcs) < cw) if fg else True

    def tor(rr):
        return rr if top else H - 1 - rr

    def toc(cc):
        return cc if left else W - 1 - cc

    # ---- 1. draw each mark-pair's two rays inside the seed quadrant ----------
    ub = min(H, W) // 4
    for k in range(max(0, ub - 1)):
        a, b = 2 * k + 1, 2 * k + 3          # corner-canonical coordinates
        col = int(I[tor(a), toc(b)])
        if col == bgc:
            continue                          # this ring carries no ray
        # ray running along row `a`, starting one step past its own mark
        hcells = [(tor(a), toc(c))
                  for c in range(b + 2, min(cw, W - 1 - b) + 1, 2)]
        if hcells:
            ops.append(col)
            sels.append(sel_of(hcells))
            for (r, c) in hcells:
                g[r, c] = col
        # ray running down column `a`
        vcells = [(tor(r), toc(a))
                  for r in range(b + 2, min(ch, H - 1 - b) + 1, 2)]
        if vcells:
            ops.append(col)
            sels.append(sel_of(vcells))
            for (r, c) in vcells:
                g[r, c] = col

    # ---- 2. mirror the quadrant across the centre column --------------------
    r0 = 0 if top else ch                     # the seed's row-half (centre row incl.)
    c_src = 0 if left else cw + 1
    c_dst = cw + 1 if left else 0

    # full rectangle: whole seed half-block, background included -> bbox is exact
    ops.append(29); sels.append([r0, c_src, ch, cw - 1])          # CopyO
    ops.append(30); sels.append([r0, c_dst, 0, 0])                # Paste

    src = g[r0:r0 + ch + 1, c_src:c_src + cw].copy()
    dst = g[r0:r0 + ch + 1, c_dst:c_dst + cw]
    repair = []
    for i in range(ch + 1):
        for j in range(cw):
            if src[i, j] != 0:
                dst[i, j] = src[i, j]
            elif dst[i, j] != 0:              # a 0-coloured cell did not travel
                repair.append((r0 + i, c_dst + j))
    if repair:
        ops.append(0); sels.append(sel_of(repair))                # Color0
        for (r, c) in repair:
            g[r, c] = 0

    # full rectangle: the pasted block is mirrored in place, background included
    ops.append(26); sels.append([r0, c_dst, ch, cw - 1])          # FlipH
    g[r0:r0 + ch + 1, c_dst:c_dst + cw] = np.fliplr(g[r0:r0 + ch + 1,
                                                      c_dst:c_dst + cw])

    # ---- 3. mirror the finished half across the centre row ------------------
    r_src = 0 if top else ch + 1
    r_dst = ch + 1 if top else 0

    ops.append(29); sels.append([r_src, 0, ch - 1, W - 1])        # CopyO
    ops.append(30); sels.append([r_dst, 0, 0, 0])                 # Paste

    src = g[r_src:r_src + ch, :].copy()
    dst = g[r_dst:r_dst + ch, :]
    repair = []
    for i in range(ch):
        for j in range(W):
            if src[i, j] != 0:
                dst[i, j] = src[i, j]
            elif dst[i, j] != 0:
                repair.append((r_dst + i, j))
    if repair:
        ops.append(0); sels.append(sel_of(repair))                # Color0
        for (r, c) in repair:
            g[r, c] = 0

    ops.append(27); sels.append([r_dst, 0, ch - 1, W - 1])        # FlipV
    g[r_dst:r_dst + ch, :] = np.flipud(g[r_dst:r_dst + ch, :])

    ops.append(34); sels.append([0, 0, H - 1, W - 1])             # Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9d9215db"
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
                                f"for task 9d9215db"
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
                    f"Failed to build a complete episode for task 9d9215db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9d9215db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset


# ── 기계적 변환 ──────────────────────────────────────────────────────────
# 원본 경로를 받아 한 자리만 바꾼다. 입력 격자도 답 격자도 보지 않고
# (ops, sels)만 읽으므로, 원본이 유도 가능하면 이것도 유도 가능하다.
import numpy as _np

_MECH_MAX = (30, 30)
_MECH_MERGEABLE = {f"Color{_i}" for _i in range(10)} | {f"FloodFill{_i}" for _i in range(10)}
_MECH_NAME = {0: 'Color0', 1: 'Color1', 2: 'Color2', 3: 'Color3', 4: 'Color4', 5: 'Color5', 6: 'Color6', 7: 'Color7', 8: 'Color8', 9: 'Color9', 10: 'FloodFill0', 11: 'FloodFill1', 12: 'FloodFill2', 13: 'FloodFill3', 14: 'FloodFill4', 15: 'FloodFill5', 16: 'FloodFill6', 17: 'FloodFill7', 18: 'FloodFill8', 19: 'FloodFill9', 20: 'MoveU', 21: 'MoveD', 22: 'MoveR', 23: 'MoveL', 24: 'Rotate90', 25: 'Rotate270', 26: 'FlipH', 27: 'FlipV', 28: 'CopyI', 29: 'CopyO', 30: 'Paste', 31: 'CopyInput', 32: 'ResetGrid', 33: 'ResizeGrid', 34: 'Submit', 35: 'None'}


def _mech_cells(sel):
    if isinstance(sel, dict) and "cells" in sel:
        return frozenset((int(r), int(c)) for r, c in sel["cells"])
    r, c, h, w = (int(v) for v in sel)        # h, w는 크기가 아니라 오프셋이다
    return frozenset((rr, cc) for rr in range(r, r + h + 1)
                     for cc in range(c, c + w + 1))


def _mech_blobs(cs):
    todo, out = set(cs), []
    while todo:
        seed = todo.pop(); comp = {seed}; stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in todo:
                        todo.discard(p); comp.add(p); stack.append(p)
        out.append(sorted(comp))
    return out


def _mech_apply(ops, sels, kind):
    n = len(ops)
    if kind == "분할":
        # 한 번에 칠하던 떨어진 덩어리들을 덩어리별로 나눠 칠한다.
        for i in range(n - 1):
            bl = _mech_blobs(_mech_cells(sels[i]))
            if len(bl) > 1:
                return (list(ops[:i]) + [ops[i]] * len(bl) + list(ops[i + 1:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in b]} for b in bl]
                        + list(sels[i + 1:]))
    else:
        # 같은 색을 연달아 두 번 칠하면 한 번에 칠한다.
        for i in range(n - 2):
            if ops[i] == ops[i + 1] and _MECH_NAME.get(int(ops[i])) in _MECH_MERGEABLE:
                u = sorted(_mech_cells(sels[i]) | _mech_cells(sels[i + 1]))
                return (list(ops[:i]) + [ops[i]] + list(ops[i + 2:]),
                        list(sels[:i]) + [{"cells": [list(p) for p in u]}]
                        + list(sels[i + 2:]))
    return None


_MECH_KIND = '병합'


def derive_operations(I, O, examples):
    ops, sels = _base_derive_operations(I, O, examples)
    out = _mech_apply(ops, sels, _MECH_KIND)
    # 변환이 걸리지 않는 인스턴스에서는 원본 경로를 그대로 쓴다.
    return out if out is not None else (ops, sels)
