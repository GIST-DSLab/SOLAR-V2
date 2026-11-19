"""
ARC Task: e21d9049 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is fixed; run colours do not affect the rule.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    from random import choice, randint

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(10))
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    ph = unifint(diff_lb, diff_ub, (2, 9))
    pw = unifint(diff_lb, diff_ub, (2, 9))
    remcols = [c for c in cols if c != bgc]
    while True:
        locih = randint(0, h - ph)
        locjh = randint(0, w - 1)
        locjw = randint(0, w - pw)
        lociw = randint(0, h - 1)
        # avoid the ambiguous case where one run abuts the other run's end in
        # the same line (the run's length could then not be read off I)
        if lociw in (locih - 1, locih + ph) and locjw <= locjh < locjw + pw:
            continue
        if locjh in (locjw - 1, locjw + pw) and locih <= lociw < locih + ph:
            continue
        break
    col = choice(remcols)
    vbar = {}
    for k in range(ph):
        r = locih + k
        vbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else choice(remcols)
    hbar = {}
    for k in range(pw):
        c = locjw + k
        hbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else choice(remcols)
    gi = [[bgc] * w for _ in range(h)]
    for (r, c), v in list(vbar.items()) + list(hbar.items()):
        gi[r][c] = v
    go = [row[:] for row in gi]
    for k in range(-(h // ph + 1), h // ph + 2):
        for (r, c), v in vbar.items():
            if 0 <= r + k * ph < h:
                go[r + k * ph][c] = v
    for k in range(-(w // pw + 1), w // pw + 2):
        for (r, c), v in hbar.items():
            if 0 <= c + k * pw < w:
                go[r][c + k * pw] = v
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Read entirely from I (O never inspected).

    One vertical run in one column and one horizontal run in one row; each
    line is tiled with its own run, period = run length, out to both edges.
    CopyI the run, Paste at every tile origin along its line; the tile cut
    off by the near edge gets a CopyI of just the run's tail.  Run cells of
    colour 0 cannot travel by paste, so those landing cells get a Color0
    right after the paste that should have carried them.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            cnt.update(np.asarray(ex[0], dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    fg = I != bgc
    ri = int(np.argmax(fg.sum(axis=1)))   # the row holding the horizontal run
    cj = int(np.argmax(fg.sum(axis=0)))   # the column holding the vertical run

    def run_of(idx):
        idx = sorted(idx)
        blocks, cur = [], [idx[0]]
        for x in idx[1:]:
            if x == cur[-1] + 1:
                cur.append(x)
            else:
                blocks.append(cur)
                cur = [x]
        blocks.append(cur)
        b = max(blocks, key=len)
        return b[0], b[-1]

    vr0, vr1 = run_of([r for r in range(h) if fg[r, cj]])
    hc0, hc1 = run_of([c for c in range(w) if fg[ri, c]])

    grid = I.copy()
    ops, sels = [], []

    def at(axis, fixed, p):
        return (p, fixed) if axis == 0 else (fixed, p)

    for axis, fixed, s0, s1, L in ((0, cj, vr0, vr1, h), (1, ri, hc0, hc1, w)):
        n = s1 - s0 + 1
        vals = [int(I[at(axis, fixed, s0 + i)]) for i in range(n)]
        base = s0 % n
        tiles = []                                # (dest, offset into run, length)
        k = base
        while k <= L - 1:
            tiles.append((k, 0, n))               # whole copies; last one clipped at the edge
            k += n
        if base > 0:
            tiles.append((0, n - base, base))     # near-edge tile shows the run's tail
        clip_src = None
        for dest, lo, ln in tiles:
            src = [at(axis, fixed, s0 + lo + i) for i in range(ln)]
            land = [(dest + i, lo + i) for i in range(ln) if dest + i <= L - 1]
            changes = [(at(axis, fixed, p), vals[j]) for p, j in land
                       if vals[j] != 0 and grid[at(axis, fixed, p)] != vals[j]]
            if changes:
                if clip_src != src:
                    ops.append(28); sels.append(sel_of(src))      # CopyI the run (or its tail)
                    clip_src = src
                ops.append(30); sels.append(sel_of([at(axis, fixed, dest)]))  # Paste at tile origin
                for cell, v in changes:
                    grid[cell] = v
            zeros = [at(axis, fixed, p) for p, j in land
                     if vals[j] == 0 and grid[at(axis, fixed, p)] != 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))         # the 0s a paste cannot carry
                for cell in zeros:
                    grid[cell] = 0

    ops.append(34); sels.append([0, 0, h - 1, w - 1])   # bbox == the whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task e21d9049"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e21d9049"
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
                                f"for task e21d9049"
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
                    f"Failed to build a complete episode for task e21d9049 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e21d9049-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
