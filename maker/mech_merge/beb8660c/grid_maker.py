"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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
    # Rule depends only on 8-line position + line lengths (color-independent),
    # so only bgc must be fixed. The DISCRETE structural variant is the final
    # rotation (which edge the 8-line lands on) -> cover all 4 in examples.
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    variants = ['identity', 'rot90', 'rot180', 'rot270']
    n_ex = num_examples if num_examples else 4
    if n_ex >= len(variants):
        examples = [{'rotf': v} for v in variants]
        examples += [{'rotf': random.choice(variants)} for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [{'rotf': v} for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {'bgc': bgc, 'instance_plan': plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rotf=None) -> dict:
    if rotf is None:
        rotf = random.choice(['identity', 'rot90', 'rot180', 'rot270'])
    cols = remove(8, interval(0, 10, 1))
    wmax = min(max_w, max_h)
    if wmax < 3:
        wmax = 3
    w = unifint(diff_lb, diff_ub, (3, wmax))
    h = unifint(diff_lb, diff_ub, (w, max_h))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        jj = h - k - 1 + j
        gi = fill(gi, c, connect((jj, 0), (jj, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rotmap = {'identity': identity, 'rot90': rot90, 'rot180': rot180, 'rot270': rot270}
    rf = rotmap[rotf]
    gi = rf(gi)
    go = rf(go)
    return {'input': gi, 'output': go}


def _base_derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = dominant color (fills most of grid)
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # locate 8-line (a full edge row/col) -> anchor edge
    edge = None
    for r in range(hi):
        if np.all(I[r, :] == 8):
            edge = 'top' if r == 0 else 'bottom'
            break
    if edge is None:
        for c in range(wi):
            if np.all(I[:, c] == 8):
                edge = 'left' if c == 0 else 'right'
                break

    horizontal = edge in ('top', 'bottom')

    # measure each line's LENGTH from I (count of its color), sort longest-first
    lengths = {c: n for c, n in cnt.items() if c != bgc and c != 8}
    order = sorted(lengths.keys(), key=lambda c: (-lengths[c], c))

    # build target: bgc canvas + 8-line + sorted lines stacked outward from 8-line,
    # aligned to the corner CCW-adjacent to the 8-edge.
    T = np.full((hi, wi), bgc, dtype=int)
    if edge == 'top':
        T[0, :] = 8
    elif edge == 'bottom':
        T[hi - 1, :] = 8
    elif edge == 'left':
        T[:, 0] = 8
    else:
        T[:, wi - 1] = 8

    new_regions = []  # (color, r, c, h, w)
    for i, color in enumerate(order):
        L = lengths[color]
        if edge == 'top':          # stack down, align left
            rr = 1 + i
            c0, c1 = 0, L - 1
            T[rr, c0:c1 + 1] = color
            new_regions.append((color, rr, c0, 0, c1 - c0))
        elif edge == 'bottom':     # stack up, align right
            rr = hi - 2 - i
            c0, c1 = wi - L, wi - 1
            T[rr, c0:c1 + 1] = color
            new_regions.append((color, rr, c0, 0, c1 - c0))
        elif edge == 'left':       # stack right, align bottom
            cc = 1 + i
            r0, r1 = hi - L, hi - 1
            T[r0:r1 + 1, cc] = color
            new_regions.append((color, r0, cc, r1 - r0, 0))
        else:                      # right: stack left, align top
            cc = wi - 2 - i
            r0, r1 = 0, L - 1
            T[r0:r1 + 1, cc] = color
            new_regions.append((color, r0, cc, r1 - r0, 0))

    # erase leftover old-line cells (old color in I, bgc in target), region by region
    erase = (I != bgc) & (I != 8) & (T == bgc)
    if horizontal:
        for r in range(hi):
            c = 0
            while c < wi:
                if erase[r, c]:
                    c0 = c
                    while c < wi and erase[r, c]:
                        c += 1
                    ops.append(int(bgc)); sels.append([r, c0, 0, c - 1 - c0])
                else:
                    c += 1
    else:
        for c in range(wi):
            r = 0
            while r < hi:
                if erase[r, c]:
                    r0 = r
                    while r < hi and erase[r, c]:
                        r += 1
                    ops.append(int(bgc)); sels.append([r0, c, r - 1 - r0, 0])
                else:
                    r += 1

    # paint each sorted line as one region
    for (color, r, c, h, w) in new_regions:
        ops.append(int(color)); sels.append([r, c, h, w])

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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
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


def derive_operations(I, O):
    ops, sels = _base_derive_operations(I, O)
    out = _mech_apply(ops, sels, _MECH_KIND)
    # 변환이 걸리지 않는 인스턴스에서는 원본 경로를 그대로 쓴다.
    return out if out is not None else (ops, sels)
