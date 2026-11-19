"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    ub = min(30, max_h, max_w)
    w = unifint(diff_lb, diff_ub, (2, min(10, ub - 1)))
    h = unifint(diff_lb, diff_ub, (w + 1, ub))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    go = replace(go, bgc, 8)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading: every corner marker shoots a ray along the long axis that bounces
    # between the two long edges.  The bounce has period P = 2*(short-1) along
    # the long axis, so the whole line pattern is ONE tile of P lines (next to
    # the markers) repeated down the grid.  Draw the tile, then stamp it.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    vert = H > W                       # long axis = rows?
    L, S = (H, W) if vert else (W, H)  # long length, short length
    P = 2 * (S - 1)

    def rc(t, s):                      # canonical (long, short) -> grid (r, c)
        return (t, s) if vert else (s, t)

    def rect(t0, t1):                  # grid bbox of long-axis band [t0, t1)
        if vert:
            return [t0, 0, t1 - t0 - 1, W - 1]
        return [0, t0, H - 1, t1 - t0 - 1]

    bseed = (1, 0) if H >= 3 else (0, 1)   # never a corner
    bgc = int(I[bseed])
    linc = [int(v) for v in np.unique(I) if int(v) != bgc][0]

    corners = sorted({(t, s) for t in (0, L - 1) for s in (0, S - 1)
                      if I[rc(t, s)] == linc})

    def tri(k):
        m = k % P
        return m if m <= S - 1 else P - m

    def ray_segments(t0, s0):
        dt = 1 if t0 == 0 else -1
        segs, seg = [], []
        for k in range(L):
            s = tri(k) if s0 == 0 else S - 1 - tri(k)
            seg.append((t0 + k * dt, s))
            if k > 0 and (s == 0 or s == S - 1):   # bounce: stroke ends here
                segs.append(seg)
                seg = [(t0 + k * dt, s)]
        if len(seg) > 1 or not segs:
            segs.append(seg)
        return segs

    ops, sels = [], []
    cur = I.copy()

    # 1) background becomes 8 (one connected region)
    ops.append(18); sels.append(sel_of([bseed]))
    cur[cur == bgc] = 8

    # 2) the tile: the P lines adjacent to the markers (whole grid if shorter)
    top = any(t == 0 for t, _ in corners)
    if L <= P:
        t_lo, t_hi = 0, L
    elif top:
        t_lo, t_hi = 0, P
    else:
        t_lo, t_hi = L - P, L
    tile_cells = []
    for (t0, s0) in corners:
        for seg in ray_segments(t0, s0):
            part = [rc(t, s) for t, s in seg if t_lo <= t < t_hi]
            tile_cells += part
            cells = [p for p in part if cur[p] != linc]
            if cells:
                ops.append(linc); sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = linc

    # 3) stamp the tile along the long axis, one period at a time
    if L > P:
        if top:
            dests = list(range(P, L, P))
        else:
            dests = list(range(L - 2 * P, -P, -P))
        clip_band = None
        for d in dests:
            a, b = max(d, 0), min(d + P, L)          # destination band
            src_a = a - d + t_lo                     # matching band in the tile
            src_b = src_a + (b - a)
            if linc != 0:
                if vert:
                    clip = cur[src_a:src_b, :].copy()
                    dst = cur[a:b, :]
                else:
                    clip = cur[:, src_a:src_b].copy()
                    dst = cur[:, a:b]
                if np.array_equal(clip, dst):
                    continue
                if clip_band != (src_a, src_b):
                    ops.append(29); sels.append(rect(src_a, src_b))   # CopyO the tile band (full rectangle)
                    clip_band = (src_a, src_b)
                ops.append(30); sels.append(list(rect(a, b)[:2]) + [0, 0])
                dst[...] = clip
            else:
                # 0 can't be pasted: paint the tile's line cells shifted by one period
                sh = d - t_lo
                cells = []
                for (r, c) in tile_cells:
                    t, s = (r, c) if vert else (c, r)
                    tt = t + sh
                    if a <= tt < b:
                        p = rc(tt, s)
                        if cur[p] != linc:
                            cells.append(p)
                if cells:
                    ops.append(linc); sels.append(sel_of(cells))
                    for p in cells:
                        cur[p] = linc

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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
