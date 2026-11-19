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
    def uf(lo, hi):
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

    if rot is None:
        rot = random.choice([0, 1, 2, 3])
    rot = int(rot) % 4

    # a quarter turn swaps the canvas dimensions, so bound the pre-rotation
    # canvas by the dimensions it will actually occupy afterwards
    if rot % 2 == 1:
        hb, wb = min(30, max_w), min(30, max_h)
    else:
        hb, wb = min(30, max_h), min(30, max_w)
    h = uf(3, max(3, hb))
    w = uf(min(10, wb), max(min(10, wb), wb))
    p = uf(2, max(2, (w - 1) // 3))
    p = max(2, min(p, max(2, (w - 1) // 3)))

    ccols = list(ccols)

    obj = {}
    for j in range(p):
        numcells = uf(1, max(1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            obj[(ii, j)] = random.choice(ccols)
    fullobj = dict(obj)
    for (i, j), c in obj.items():
        fullobj[(i, j + p)] = c
    addonw = random.randint(0, p)
    leftshift = random.randint(0, addonw)

    gi = [[bgc] * w for _ in range(h)]
    placed = {}
    for (i, j), c in fullobj.items():
        jj = j - leftshift
        if 0 <= jj < w and 0 <= i < h:
            placed[(i, jj)] = c
    for (i, j), c in placed.items():
        gi[i][j] = c

    # the answer repeats the block with its true horizontal period (which can be
    # finer than p when the sampled columns happen to agree), matching the verifier
    maxc = max(j for _, j in placed)
    per = maxc + 1
    for cand in range(1, maxc + 1):
        if all(placed.get((i, j - cand)) == v for (i, j), v in placed.items() if j - cand >= 0):
            per = cand
            break
    go = [[gi[i][j % per] for j in range(w)] for i in range(h)]

    def rot90cw(g):
        return [list(r) for r in zip(*g[::-1])]

    for _ in range(rot):
        gi = rot90cw(gi)
        go = rot90cw(go)

    return {"input": tuple(tuple(r) for r in gi), "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    """Continue the striped block across the grid.

    Every parameter -- which band repeats, along which axis, with which period, and how
    many times it still has to be stamped -- is measured from I; O is never inspected.
    """
    A = np.asarray(I, dtype=int)
    H, W = A.shape

    # 1) Orientation frame, from I alone: exactly one quarter turn puts the striped
    #    block against the bottom-left corner, i.e. leaves the top row and the right
    #    column pure background.  All measurements below live in that frame.
    kk = 0
    for k in (0, 3, 2, 1):  # identity, rot90 cw, rot180, rot90 ccw
        Ck = np.rot90(A, k)
        if len(set(Ck[0].tolist())) == 1 and len(set(Ck[:, -1].tolist())) == 1:
            kk = k
            break
    C = np.rot90(A, kk)
    Hc, Wc = C.shape
    bgc = int(C[0, -1])

    def to_orig(r, c):
        if kk == 0:
            return (r, c)
        if kk == 1:
            return (c, W - 1 - r)
        if kk == 2:
            return (H - 1 - r, W - 1 - c)
        return (H - 1 - c, r)

    def orig_rect(r0, c0, r1, c1):
        pts = [to_orig(r0, c0), to_orig(r0, c1), to_orig(r1, c0), to_orig(r1, c1)]
        rs = [q[0] for q in pts]
        cs = [q[1] for q in pts]
        return [min(rs), min(cs), max(rs) - min(rs), max(cs) - min(cs)]

    cells = np.argwhere(C != bgc)
    if len(cells) == 0:
        return [34], [[0, 0, H - 1, W - 1]]

    # 2) the block and the period with which its columns repeat
    minc = int(cells[:, 1].min())
    maxc = int(cells[:, 1].max())
    colors = {(int(r), int(c)): int(C[r, c]) for r, c in cells}
    wobj = maxc - minc + 1
    p = wobj
    for cand in range(1, wobj):
        if all(colors.get((r, c - cand)) == v for (r, c), v in colors.items() if c - cand >= minc):
            p = cand
            break

    # 3) what continuing the block means: every band of p columns from the block onwards
    #    repeats the first period.  (Target kept only to drop stamps that change nothing.)
    T = C.copy()
    for c in range(minc, Wc):
        T[:, c] = C[:, minc + (c - minc) % p]
    target = np.rot90(T, -kk)

    src = orig_rect(0, minc, Hc - 1, minc + p - 1)  # full rectangle: one period of the band

    def run(entries):
        g = A.copy()
        clip = None
        for op, sel in entries:
            if op == 28:
                r, c, h, w = sel
                clip = A[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                if clip is None:
                    continue
                r, c = sel[0], sel[1]
                ch, cw = clip.shape
                r1, c1 = min(H, r + ch), min(W, c + cw)
                sub = clip[:r1 - r, :c1 - c]
                reg = g[r:r1, c:c1]
                g[r:r1, c:c1] = np.where(sub != 0, sub, reg)
            else:
                for (rr, cc) in sel:
                    g[rr, cc] = op
        return g

    # 4) stamp the period across the grid, band by band, skipping any band that already
    #    holds it (the block itself is often two periods wide)
    entries = [(28, src)]
    k = 1
    while minc + k * p < Wc:
        c0 = minc + k * p
        wd = min(p, Wc - c0)
        dr = orig_rect(0, c0, Hc - 1, c0 + wd - 1)
        origin = [dr[0], dr[1], 0, 0]
        before = run(entries)
        cand = entries + [(30, origin)]
        after = run(cand)
        if not np.array_equal(after, before):
            # a band that only partly fits is simply clipped when it runs off the far
            # edge; when it runs off the near edge the stamp has to start there, so
            # re-copy the part of the period that fits
            if wd < p and not np.array_equal(after[target != 0], target[target != 0]):
                sub_src = orig_rect(0, minc, Hc - 1, minc + wd - 1)
                cand = entries + [(28, sub_src), (30, origin)]
                after = run(cand)
            if not np.array_equal(after, before):
                entries = cand
        k += 1

    # 5) copy/paste cannot carry colour 0, so wherever the repeated band is black the
    #    stamps left the background standing: paint those cells black in one go
    work = run(entries)
    black = []
    for r in range(Hc):
        for t in range(p):
            if int(C[r, minc + t]) != 0:
                continue
            c = minc + t + p
            while c < Wc:
                rr, cc = to_orig(r, c)
                if work[rr, cc] != 0:
                    black.append((rr, cc))
                c += p
    if black:
        entries.append((0, sorted(black)))

    # 6) drop anything the grid does not need
    changed = True
    while changed:
        changed = False
        for i in range(len(entries)):
            trimmed = entries[:i] + entries[i + 1:]
            if np.array_equal(run(trimmed), target):
                entries = trimmed
                changed = True
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
