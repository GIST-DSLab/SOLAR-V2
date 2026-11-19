"""
ARC Task: 6773b310 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 1]
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, linc: int, fgc: int) -> dict:
    # block size / block count bounded so that n*(s+1)-1 <= max_dim
    smax_h = min(5, (max_h + 1) // 2 - 1)
    if smax_h < 2:
        smax_h = 2
    h = unifint(diff_lb, diff_ub, (2, smax_h))
    nhmax = min(5, (max_h + 1) // (h + 1))
    if nhmax < 2:
        nhmax = 2
    nh = unifint(diff_lb, diff_ub, (2, nhmax))

    smax_w = min(5, (max_w + 1) // 2 - 1)
    if smax_w < 2:
        smax_w = 2
    w = unifint(diff_lb, diff_ub, (2, smax_w))
    nwmax = min(5, (max_w + 1) // (w + 1))
    if nwmax < 2:
        nwmax = 2
    nw = unifint(diff_lb, diff_ub, (2, nwmax))

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    nbldev = unifint(diff_lb, diff_ub, (0, (nh * nw) // 2))
    nbl = choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = sample(llocs, nbl)
    bglocs = difference(llocs, bluelocs)
    inds = totuple(asindices(smallc))
    gi = tuple(e for e in c)
    go = canvas(bgc, (nh, nw))
    for ij in bluelocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 2)))
        gi = paint(gi, shift(subg, ij))
        a, b = ij
        loci = a // (h + 1)
        locj = b // (w + 1)
        go = fill(go, 1, {(loci, locj)})
    for ij in bglocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 1)))
        gi = paint(gi, shift(subg, ij))
    return {'input': gi, 'output': go}


def _base_derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # ---- read the lattice out of I -------------------------------------
    sep_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    linc = int(I[sep_rows[0], 0])
    nh = len(sep_rows) + 1
    nw = len(sep_cols) + 1
    bh = sep_rows[0]          # block height
    bw = sep_cols[0]          # block width

    # ---- read the two non-line colors out of I -------------------------
    cnt = Counter(I.flatten().tolist())
    if linc in cnt:
        del cnt[linc]
    fgc = min(cnt.items(), key=lambda kv: kv[1])[0]      # rarer -> markers
    bgc = [c for c in cnt if c != fgc][0]

    # ---- count markers inside every lattice cell of I -------------------
    counts = np.zeros((nh, nw), dtype=int)
    for i in range(nh):
        for j in range(nw):
            r0, c0 = i * (bh + 1), j * (bw + 1)
            counts[i, j] = int((I[r0:r0 + bh, c0:c0 + bw] == fgc).sum())
    mx = int(counts.max())
    mask = (counts == mx)                                # -> blue in O

    # ---- region decomposition helper -----------------------------------
    def regions(m):
        H, W = m.shape
        m = m.copy()
        comps = []
        for sr in range(H):
            for sc in range(W):
                if not m[sr, sc]:
                    continue
                comp, stack = set(), [(sr, sc)]
                while stack:
                    r, c = stack.pop()
                    if not (0 <= r < H and 0 <= c < W) or (r, c) in comp:
                        continue
                    if not m[r, c]:
                        continue
                    comp.add((r, c))
                    stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                for (r, c) in comp:
                    m[r, c] = False
                comps.append(comp)
        comps.sort(key=len, reverse=True)
        rects = []
        for comp in comps:
            rem = set(comp)
            while rem:
                r, c = min(rem)
                ww = 0
                while (r, c + ww + 1) in rem:
                    ww += 1
                hh = 0
                while all((r + hh + 1, cc) in rem for cc in range(c, c + ww + 1)):
                    hh += 1
                rects.append([r, c, hh, ww])
                for rr in range(r, r + hh + 1):
                    for cc in range(c, c + ww + 1):
                        rem.discard((rr, cc))
        return rects

    ops, sels = [], []
    # shrink the canvas to one cell per lattice cell
    ops.append(33); sels.append([0, 0, nh - 1, nw - 1])

    # paint each blue region (these cells never already hold 1)
    for rect in regions(mask):
        ops.append(1); sels.append(rect)

    # clear the leftover markers/lines that survived the crop
    stray = np.zeros((nh, nw), dtype=bool)
    for i in range(nh):
        for j in range(nw):
            if not mask[i, j] and int(I[i, j]) != bgc:
                stray[i, j] = True
    for rect in regions(stray):
        ops.append(int(bgc)); sels.append(rect)

    ops.append(34); sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6773b310"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6773b310"
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
                                f"for task 6773b310"
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
                    f"Failed to build a complete episode for task 6773b310 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6773b310-rearc-llm_{_sn + 1}",
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
