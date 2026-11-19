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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))
    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(15, min(max_h, max_w) // 2))
    h = unifint(diff_lb, diff_ub, (3, hmax))
    w = h
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    g = np.full((h, w), bgc, dtype=int)
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
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
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx = bx - {ch}
    for (i, j), v in obj.items():
        g[i, j] = v
    g2 = g.copy()
    dm = g.T
    for i in range(h):
        for j in range(w):
            if j >= i:
                g2[i, j] = dm[i, j]
    g = g2
    g = np.hstack([g, g[:, ::-1]])
    g = np.vstack([g, g[::-1, :]])
    go = g.copy()
    gi = g.copy()
    for alph in (2, 1):
        locidev = unifint(diff_lb, diff_ub, (1, alph * h))
        locjdev = unifint(diff_lb, diff_ub, (1, w))
        loci = alph * h - locidev
        locj = w - locjdev
        loci2 = unifint(diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
        gi[loci:loci2 + 1, locj:locj2 + 1] = 4
        gi = gi[::-1, ::-1].copy()
        go = go[::-1, ::-1].copy()
    mfs = [lambda x: x, lambda x: x.T, lambda x: x[::-1, ::-1].T,
           lambda x: x[:, ::-1], lambda x: x[::-1, :],
           lambda x: np.rot90(x, 3), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    # How the golf entry reads the task: the picture equals its own left-right mirror,
    # so the first unit is the WHOLE GRID. Flipping the whole picture fills every
    # 4-patch with its mirror partner. The flip also carries the 4s to the mirror side.
    # The golf entry fills those leftover 4s from the OTHER mirror (up-down), so here
    # each leftover 4-rectangle gets a copy of its up-down partner from the input,
    # which is then flipped up-down in place. If that partner is also covered, the
    # fallback is the same block in the input, then the point-mirror block.
    # Where this differs from the maker: the maker repairs each patch separately with
    # a block copy. Here one whole-picture flip does most of the work first.
    # Nothing here is read from O.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    whole = [0, 0, H - 1, W - 1]                  # the whole picture, a full rectangle
    ops, sels = [], []

    if not (I == 4).any():
        ops.append(34); sels.append(whole)
        return ops, sels

    # 1. Flip the whole grid left-right. If the 4-set is itself left-right
    #    symmetric, that flip would change nothing, so flip up-down instead.
    if not np.array_equal(np.fliplr(I), I):
        other = 'ud'
        ops.append(26); sels.append(whole)
        G = np.fliplr(I).copy()
    else:
        other = 'lr'
        ops.append(27); sels.append(whole)
        G = np.flipud(I).copy()

    def part(axis, r0, c0, r1, c1):
        if axis == 'lr':
            return r0, W - 1 - c1
        if axis == 'ud':
            return H - 1 - r1, c0
        if axis == 'rot':
            return H - 1 - r1, W - 1 - c1
        return r0, c0                              # 'same': this block of the input

    FLIPS = {'lr': ('lr',), 'ud': ('ud',), 'rot': ('lr', 'ud'), 'same': ()}

    def fill(r0, c0, r1, c1):
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        for axis in (other, 'same', 'rot'):
            sr, sc = part(axis, r0, c0, r1, c1)
            src = I[sr:sr + bh, sc:sc + bw]
            if (src == 4).any():
                continue
            src = src.copy()
            if (src == 0).any():                   # Paste skips 0, so clear the patch first
                ops.append(0); sels.append([r0, c0, bh - 1, bw - 1])
                G[r0:r1 + 1, c0:c1 + 1] = 0
            if (src != 0).any():
                ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])   # CopyI the partner block
                ops.append(30); sels.append([r0, c0, 0, 0])             # Paste onto the patch
                blk = G[r0:r1 + 1, c0:c1 + 1]
                blk[src != 0] = src[src != 0]
            cur = G[r0:r1 + 1, c0:c1 + 1].copy()
            for f in FLIPS[axis]:
                nxt = np.fliplr(cur) if f == 'lr' else np.flipud(cur)
                if not np.array_equal(nxt, cur):
                    ops.append(26 if f == 'lr' else 27)
                    sels.append([r0, c0, bh - 1, bw - 1])   # whole patch rectangle
                    cur = nxt
            G[r0:r1 + 1, c0:c1 + 1] = cur
            return True
        return False

    def fill_split(r0, c0, r1, c1):
        if fill(r0, c0, r1, c1):
            return
        if r1 > r0:
            for r in range(r0, r1 + 1):
                fill_split(r, c0, r, c1)
        elif c1 > c0:
            for c in range(c0, c1 + 1):
                fill_split(r0, c, r0, c)

    # 2. The 4s the flip carried across: split them into whole rectangles and fill each one.
    m = (G == 4)
    while m.any():
        rs, cs = np.nonzero(m)
        r0, c0 = int(rs[0]), int(cs[0])
        c1 = c0
        while c1 + 1 < W and m[r0, c1 + 1]:
            c1 += 1
        r1 = r0
        while r1 + 1 < H and m[r1 + 1, c0:c1 + 1].all():
            r1 += 1
        m[r0:r1 + 1, c0:c1 + 1] = False
        fill_split(r0, c0, r1, c1)

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
