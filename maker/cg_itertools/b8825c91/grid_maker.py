"""
ARC Task: b8825c91 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The rule (hidden 4-patches restored from symmetric partners) is colour-agnostic
    # apart from the fixed mask colour 4; only the background is an episode role.
    return {"bgc": random.choice([c for c in range(10) if c != 4])}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    from random import choice, sample

    def uni(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = choice(cols)
    h = uni(3, max(3, min(15, min(max_h, max_w) // 2)))
    w = h
    remcols = [c for c in cols if c != bgc]
    remcols = sample(remcols, uni(1, 8))
    g = np.full((h, w), bgc, dtype=int)
    nc = uni(1, h * w)
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = choice(sorted(bx))
    obj = {start: choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = choice(cand)
        obj[ch] = choice(remcols)
        bx.discard(ch)
    for (i, j), c in obj.items():
        g[i, j] = c
    for i in range(h):
        for j in range(w):
            if j >= i:
                g[j, i] = g[i, j]
    g = np.hstack([g, np.fliplr(g)])
    g = np.vstack([g, np.flipud(g)])
    gi = g.copy()
    go = g.copy()
    for alph in (2, 1):
        loci = alph * h - uni(1, alph * h)
        locj = w - uni(1, w)
        loci2 = uni(loci, alph * h - 1)
        locj2 = uni(locj, w - 1)
        gi[loci:loci2 + 1, locj:locj2 + 1] = 4
        gi, go = np.rot90(gi, 2), np.rot90(go, 2)
    mfs = [lambda x: x, lambda x: x.T, lambda x: np.rot90(x, 2).T, np.fliplr, np.flipud,
           lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1)]
    for fn in sample(mfs, choice((1, 2))):
        gi, go = fn(gi), fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the picture is point-symmetric through its
    # centre, so every hidden cell (r, c) equals its 180-degree partner (H-1-r, W-1-c).
    # Each rectangular 4-patch is rebuilt by copying its point-reflected partner rectangle
    # and flipping it both ways in place.  Only where that partner is itself hidden do we
    # fall back to the left-right / up-down mirror partners (the picture has those too).
    # Everything is read from I; O is never consulted.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    G = I.copy()
    touched = np.zeros((H, W), dtype=bool)
    ops, sels = [], []

    def rects_of(mask, mode):
        m = mask.copy()
        out = []
        while m.any():
            rs, cs = np.nonzero(m)
            r0, c0 = int(rs[0]), int(cs[0])
            if mode == 'cell':
                out.append((r0, c0, r0, c0))
                m[r0, c0] = False
                continue
            c1 = c0
            while c1 + 1 < W and m[r0, c1 + 1]:
                c1 += 1
            r1 = r0
            if mode == 'rect':
                while r1 + 1 < H and m[r1 + 1, c0:c1 + 1].all():
                    r1 += 1
            out.append((r0, c0, r1, c1))
            m[r0:r1 + 1, c0:c1 + 1] = False
        return out

    AXES = ('rot', 'lr', 'ud')   # point symmetry first

    def source_of(axis, r0, c0, r1, c1):
        if axis == 'rot':
            return H - 1 - r1, W - 1 - c1
        if axis == 'lr':
            return r0, W - 1 - c1
        return H - 1 - r1, c0

    for mode in ('rect', 'row', 'cell'):
        remaining = rects_of(G == 4, mode)
        while remaining:
            progressed = False
            for blk in list(remaining):
                r0, c0, r1, c1 = blk
                bh, bw = r1 - r0 + 1, c1 - c0 + 1
                cands = []
                for k, axis in enumerate(AXES):
                    sr, sc = source_of(axis, r0, c0, r1, c1)
                    if (G[sr:sr + bh, sc:sc + bw] == 4).any():
                        continue
                    fresh = not touched[sr:sr + bh, sc:sc + bw].any()
                    cands.append((0 if fresh else 1, k, axis, sr, sc))
                if not cands:
                    continue
                cands.sort()
                fr, _, axis, sr, sc = cands[0]
                src = G[sr:sr + bh, sc:sc + bw].copy()
                # copy the full partner rectangle (from I if untouched, else working grid)
                ops.append(28 if fr == 0 else 29)
                sels.append([sr, sc, bh - 1, bw - 1])
                # Paste skips 0s: clear the whole patch rectangle first if 0s must land
                if (src == 0).any():
                    ops.append(0)
                    sels.append([r0, c0, bh - 1, bw - 1])
                ops.append(30)
                sels.append([r0, c0, 0, 0])
                cur = src
                if axis in ('lr', 'rot'):
                    f = np.fliplr(cur)
                    if not np.array_equal(f, cur):
                        ops.append(26)                       # whole patch rectangle
                        sels.append([r0, c0, bh - 1, bw - 1])
                    cur = f
                if axis in ('ud', 'rot'):
                    f = np.flipud(cur)
                    if not np.array_equal(f, cur):
                        ops.append(27)                       # whole patch rectangle
                        sels.append([r0, c0, bh - 1, bw - 1])
                    cur = f
                G[r0:r1 + 1, c0:c1 + 1] = cur
                touched[r0:r1 + 1, c0:c1 + 1] = True
                remaining.remove(blk)
                progressed = True
            if not progressed:
                break
        if not (G == 4).any():
            break

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
                        f"num_examples+1 ({num_examples + 1}) for task b8825c91"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8825c91"
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
                                f"for task b8825c91"
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
                    f"Failed to build a complete episode for task b8825c91 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8825c91-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
