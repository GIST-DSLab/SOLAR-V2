"""
ARC Task: d8c310e9 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


# the generator turns the finished pair by a quarter turn, so the block can march
# right / up / left / down: plan all four so every case is shown in the examples
VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numc = random.randint(1, 9)
    ccols = random.sample(remcols, numc)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": list(ccols), "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, rot=None) -> dict:
    def unif(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi < lo:
            hi = lo
        a = lo + int((hi - lo) * diff_lb)
        b = lo + int((hi - lo) * diff_ub)
        a = max(lo, min(hi, a))
        b = max(lo, min(hi, b))
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    def _canonical_frame(A):
        """the quarter turn that stands the block in the bottom-left corner"""
        for k in (0, 3, 2, 1):  # identity, rot90 cw, rot180, rot90 ccw
            C = np.rot90(A, k)
            if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
                return k, C
        return None, None

    def _extend(A):
        """the rule: stamp the block along its axis every period, all the way across"""
        k, C = _canonical_frame(A)
        if k is None:
            return A.copy()
        bgc = int(C[0, -1])
        cells = np.argwhere(C != bgc)
        if len(cells) == 0:
            return A.copy()
        colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
        minc, maxc = int(cells[:, 1].min()), int(cells[:, 1].max())
        per = maxc - minc + 1
        for cand in range(1, maxc - minc + 1):
            if all(colors.get((r, c - cand)) == v for (r, c), v in colors.items() if c - cand >= minc):
                per = cand
                break
        T = C.copy()
        for off in range(0, C.shape[1] + 1, per):
            for (r, c), v in colors.items():
                if c + off < C.shape[1]:
                    T[r, c + off] = v
        return np.rot90(T, -k)

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    rot = int(rot) % 4

    # a quarter turn swaps the canvas dimensions, so bound the pre-rotation canvas by
    # the dimensions it is going to occupy after the turn
    if rot % 2 == 1:
        hb, wb = min(30, max_w), min(30, max_h)
    else:
        hb, wb = min(30, max_h), min(30, max_w)
    ccols = list(ccols)

    A = B = None
    for _ in range(64):
        h = unif(3, max(3, hb))
        w = unif(min(10, wb), max(min(10, wb), wb))
        p = unif(2, max(2, (w - 1) // 3))
        p = max(2, min(p, max(2, (w - 1) // 3)))

        # a bottom-anchored strip p columns wide, repeated once p columns to its right,
        # then slid left so the visible block may start mid-period
        obj = {}
        for j in range(p):
            numcells = unif(1, max(1, h - 1))
            for ii in range(h - 1, h - numcells - 1, -1):
                obj[(ii, j)] = random.choice(ccols)
        fullobj = dict(obj)
        for (i, j), c in obj.items():
            fullobj[(i, j + p)] = c
        addonw = random.randint(0, p)
        leftshift = random.randint(0, addonw)

        gi = [[bgc] * w for _ in range(h)]
        for (i, j), c in fullobj.items():
            jj = j - leftshift
            if 0 <= i < h and 0 <= jj < w:
                gi[i][jj] = c
        A = np.rot90(np.array(gi, dtype=int), -rot)  # np.rot90(., -1) is a clockwise turn
        B = _extend(A)
        if not np.array_equal(A, B):
            break

    return {"input": tuple(map(tuple, A.tolist())), "output": tuple(map(tuple, B.tolist()))}


def derive_operations(I, O):
    """Continue the striped block until it fills the grid.

    Which block repeats, along which axis, with which period, and how many stamps are
    still missing are all measured from I; O is never looked at.
    """
    A = np.asarray(I, dtype=int)
    H, W = A.shape

    # 1) frame: exactly one quarter turn leaves the top row and the right column pure
    #    background, i.e. stands the block in the bottom-left corner.  That turn also
    #    says which way the block marches across the grid as it is given.
    axis = {0: (0, 1), 3: (-1, 0), 2: (0, -1), 1: (1, 0)}
    kk = None
    for k in (0, 3, 2, 1):  # identity, rot90 cw, rot180, rot90 ccw
        C = np.rot90(A, k)
        if len(set(C[0].tolist())) == 1 and len(set(C[:, -1].tolist())) == 1:
            kk = k
            break
    if kk is None:
        return [34], [[0, 0, H - 1, W - 1]]
    C = np.rot90(A, kk)
    bgc = int(C[0, -1])
    dr, dc = axis[kk]

    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [[0, 0, H - 1, W - 1]]

    # 2) period: the shortest slide along that axis that drops the block onto itself
    colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
    minc = int(cells[:, 1].min())
    maxc = int(cells[:, 1].max())
    per = maxc - minc + 1
    for cand in range(1, maxc - minc + 1):
        if all(colors.get((r, c - cand)) == v for (r, c), v in colors.items() if c - cand >= minc):
            per = cand
            break

    # 3) the block as a rectangle of the grid as it is given
    ocells = np.argwhere(A != bgc)
    r0, r1 = int(ocells[:, 0].min()), int(ocells[:, 0].max())
    c0, c1 = int(ocells[:, 1].min()), int(ocells[:, 1].max())

    def run(entries):
        g = A.copy()
        clip = None
        for op, sel in entries:
            if op == 28:
                a, b, hh, ww = sel
                clip = A[a:a + hh + 1, b:b + ww + 1]
            elif op == 30:
                if clip is None:
                    continue
                a, b = sel[0], sel[1]
                ch, cw = clip.shape
                a1, b1 = min(H, a + ch), min(W, b + cw)
                sub = clip[:a1 - a, :b1 - b]
                reg = g[a:a1, b:b1]
                g[a:a1, b:b1] = np.where(sub != 0, sub, reg)
            else:
                for (rr, cc) in sel:
                    g[rr, cc] = op
        return g

    entries = []
    grid = A.copy()
    clip_rect = None
    step = 1
    while True:
        orr, occ = dr * per * step, dc * per * step
        tr0, tr1 = r0 + orr, r1 + orr
        tc0, tc1 = c0 + occ, c1 + occ
        if tr1 < 0 or tr0 >= H or tc1 < 0 or tc0 >= W:
            break
        # a stamp running off the far edge is simply clipped; one running off the near
        # edge has to start at the edge, so it carries only the part of the block that fits
        dr0, dc0 = max(0, tr0), max(0, tc0)
        sr0, sc0 = dr0 - orr, dc0 - occ
        rect = (sr0, sc0, r1 - sr0, c1 - sc0)
        src = A[sr0:r1 + 1, sc0:c1 + 1]
        sh, sw = min(src.shape[0], H - dr0), min(src.shape[1], W - dc0)

        after = grid.copy()
        reg = after[dr0:dr0 + sh, dc0:dc0 + sw]
        cut = src[:sh, :sw]
        after[dr0:dr0 + sh, dc0:dc0 + sw] = np.where(cut != 0, cut, reg)
        if not np.array_equal(after, grid):
            if clip_rect != rect:
                entries.append((28, list(rect)))  # full rectangle: the block itself
                clip_rect = rect
            entries.append((30, [dr0, dc0, 0, 0]))
            grid = after

        # the clipboard treats colour 0 as empty, so wherever the block is black the
        # stamp left the background standing: black those cells in one go
        if bgc != 0:
            black = [(dr0 + i, dc0 + j)
                     for i in range(sh) for j in range(sw)
                     if cut[i, j] == 0 and grid[dr0 + i, dc0 + j] != 0]
            if black:
                entries.append((0, black))
                for (rr, cc) in black:
                    grid[rr, cc] = 0
        step += 1

    # the block is several periods wide, so consecutive stamps overlap: keep only the
    # stamps the finished grid actually needs
    final = run(entries)
    dropped = True
    while dropped:
        dropped = False
        for i in range(len(entries)):
            trimmed = entries[:i] + entries[i + 1:]
            if np.array_equal(run(trimmed), final):
                entries = trimmed
                dropped = True
                break

    ops, sels = [], []
    for op, sel in entries:
        ops.append(op)
        sels.append(sel_of(sel) if op == 0 else sel)
    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d8c310e9"
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
                                f"for task d8c310e9"
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
                    f"Failed to build a complete episode for task d8c310e9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d8c310e9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
