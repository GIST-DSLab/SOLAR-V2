"""
ARC Task: 469497ad (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c != 2]
    bgc, sqc = random.sample(cols, 2)
    rest = [c for c in cols if c not in (bgc, sqc)]
    random.shuffle(rest)
    return {"bgc": bgc, "sqc": sqc, "acc_cols": rest}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, sqc=None, acc_cols=None) -> dict:
    from dsl import (canvas, backdrop, fill, rot180, shoot, upscale, ofcolor, lrcorner,
                     ulcorner, urcorner, llcorner, identity, rot90, rot270, interval)
    from utils import unifint
    cols = [c for c in range(10) if c != 2]
    if bgc is None or sqc is None:
        bgc, sqc = random.sample(cols, 2)
    if acc_cols is None:
        acc_cols = [c for c in cols if c not in (bgc, sqc)]
        random.shuffle(acc_cols)
    lim = min(max_h, max_w)
    hub = max(3, min(6, lim // 2))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = unifint(diff_lb, diff_ub, (3, hub))
    gi = canvas(bgc, (h, w))
    sqh = random.randint(1, h - 2)
    sqw = random.randint(1, w - 2)
    sqloci = random.randint(0, h - sqh - 2)
    sqlocj = random.randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(h, w) + 1, lim // max(h, w), 7)
    if numcub < 2:
        raise ValueError("grid too small")
    numc = unifint(diff_lb, diff_ub, (2, numcub))
    numaccc = numc - 1
    ccols = list(acc_cols[:numaccc])
    gi = rot180(gi)
    locs = random.sample(interval(1, min(h, w), 1), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    gi = rot180(gi)
    go = upscale(gi, numc)
    rect = ofcolor(go, sqc)
    ll = (shoot(lrcorner(rect), (1, 1)) | shoot(ulcorner(rect), (-1, -1)) |
          shoot(urcorner(rect), (-1, 1)) | shoot(llcorner(rect), (1, -1)))
    go = fill(go, 2, ll & ofcolor(go, bgc))
    rotf = random.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O):
    # Reading (as in the golf entry): stretch every ROW k times, then every COLUMN
    # k times (k = numcolors - 1), then grow diagonal rays out of the square's corners.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    palette = sorted(set(I.flatten().tolist()))
    k = len(palette) - 1                     # stretch factor = number of colours - 1
    ho, wo = hi * k, wi * k

    # --- the square: a colour whose surrounding ring is a single colour (the background)
    def ring(r0, c0, r1, c1):
        out = set()
        for r in range(r0 - 1, r1 + 2):
            for c in range(c0 - 1, c1 + 2):
                if (r in (r0 - 1, r1 + 1) or c in (c0 - 1, c1 + 1)) and 0 <= r < hi and 0 <= c < wi:
                    out.add(int(I[r, c]))
        return out

    best = None
    for col in palette:
        cells = np.argwhere(I == col)
        r0, c0 = [int(v) for v in cells.min(0)]
        r1, c1 = [int(v) for v in cells.max(0)]
        rg = ring(r0, c0, r1, c1)
        if len(rg) == 1:
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if best is None or area < best[0]:
                best = (area, (r0, c0, r1, c1), next(iter(rg)))
    _, (sr0, sc0, sr1, sc1), bgc = best

    ops, sels = [], []
    G = np.zeros((ho, wo), dtype=int)       # mirror of the working grid
    G[:hi, :wi] = I

    # 1. grow the canvas to the upscaled size (input stays at the top-left)
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])

    clipboard = {"held": None, "pending": None}

    def paste_band(clip, r, c):
        """Paste a band (row or column strip) at (r, c); clear the stale band first if
        the clip's 0-cells would otherwise leave old content showing through."""
        bh, bw = clip.shape
        dest = G[r:r + bh, c:c + bw]
        if np.array_equal(dest, clip):
            return
        need_clear = bool(((clip == 0) & (dest != 0)).any())
        need_paste = bool((clip != 0).any())
        if need_paste:
            held = clipboard["held"]
            if held is None or held.shape != clip.shape or not np.array_equal(held, clip):
                op, sel = clipboard["pending"]                   # copy only when needed
                ops.append(op); sels.append(sel)
                clipboard["held"] = clip
        if need_clear:
            ops.append(0); sels.append([r, c, bh - 1, bw - 1])   # whole destination band
            G[r:r + bh, c:c + bw] = 0
        if need_paste:
            ops.append(30); sels.append([r, c, 0, 0])
            dest = G[r:r + bh, c:c + bw]
            G[r:r + bh, c:c + bw] = np.where(clip != 0, clip, dest)

    # 2. stretch the rows: each input row i is laid down k times, bottom row first
    for i in range(hi - 1, -1, -1):
        clip = I[i:i + 1, :].copy()
        clipboard["pending"] = (28, [i, 0, 0, wi - 1])           # CopyI input row i
        for t in range(k):
            paste_band(clip, i * k + t, 0)

    # 3. stretch the columns: each column j of the row-stretched grid repeated k times,
    #    rightmost column first so no source column is overwritten before it is copied
    for j in range(wi - 1, -1, -1):
        clip = G[:, j:j + 1].copy()
        clipboard["pending"] = (29, [0, j, ho - 1, 0])           # CopyO column j
        for t in range(k):
            paste_band(clip, 0, j * k + t)

    # 4. shoot a diagonal ray from each corner of the stretched square over the background
    rr0, cc0 = sr0 * k, sc0 * k
    rr1, cc1 = (sr1 + 1) * k - 1, (sc1 + 1) * k - 1
    rays = [((rr0, cc0), (-1, -1)), ((rr0, cc1), (-1, 1)),
            ((rr1, cc1), (1, 1)), ((rr1, cc0), (1, -1))]
    for (r, c), (dr, dc) in rays:
        cells = []
        r += dr; c += dc
        while 0 <= r < ho and 0 <= c < wo:
            if G[r, c] == bgc:
                cells.append((r, c))
            r += dr; c += dc
        if cells:
            ops.append(2); sels.append(sel_of(cells))
            for (a, b) in cells:
                G[a, b] = 2

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
                        f"num_examples+1 ({num_examples + 1}) for task 469497ad"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 469497ad"
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
                                f"for task 469497ad"
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
                    f"Failed to build a complete episode for task 469497ad "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"469497ad-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
