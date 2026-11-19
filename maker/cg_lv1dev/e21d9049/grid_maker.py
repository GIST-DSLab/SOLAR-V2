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
    # Only the background is an episode-level colour role.  The run colours are
    # a pattern that gets copied as-is; the rule never chooses a colour.
    return {"bgc": random.choice(range(10))}


def _bg(I):
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def _runs(I, bgc):
    """Return every maximal straight segment of non-background cells that is at
    least 2 long, as (vertical_runs, horizontal_runs) of (line, start, end)."""
    h, w = I.shape
    fg = I != bgc
    vr, hr = [], []
    for c in range(w):
        r = 0
        while r < h:
            if fg[r, c]:
                s = r
                while r + 1 < h and fg[r + 1, c]:
                    r += 1
                if r > s:
                    vr.append((c, s, r))
            r += 1
    for r in range(h):
        c = 0
        while c < w:
            if fg[r, c]:
                s = c
                while c + 1 < w and fg[r, c + 1]:
                    c += 1
                if c > s:
                    hr.append((r, s, c))
            c += 1
    return vr, hr


def _read_runs(I):
    I = np.asarray(I, dtype=int)
    bgc = _bg(I)
    vr, hr = _runs(I, bgc)
    if len(vr) != 1 or len(hr) != 1:
        return None
    (cj, v0, v1), (ri, h0, h1) = vr[0], hr[0]
    return bgc, cj, v0, v1, ri, h0, h1


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lo, hi):
        lo, hi = min(lo, hi), max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    if bgc is None:
        bgc = random.randint(0, 9)
    remcols = [c for c in range(10) if c != bgc]
    for _ in range(1000):
        h = unifint(10, max(10, max_h))
        w = unifint(10, max(10, max_w))
        ph = unifint(2, 9)
        pw = unifint(2, 9)
        hbar = {(k, 0): random.choice(remcols) for k in range(ph)}
        wbar = {(0, k): random.choice(remcols) for k in range(pw)}
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        hbar = {(r + locih, c + locjh): v for (r, c), v in hbar.items()}
        wbar = {(r + lociw, c + locjw): v for (r, c), v in wbar.items()}
        col = random.choice(remcols)
        for (r, c) in list(hbar):
            if abs(r - lociw) % ph == 0:
                hbar[(r, c)] = col
        for (r, c) in list(wbar):
            if abs(c - locjh) % pw == 0:
                wbar[(r, c)] = col
        gi = np.full((h, w), bgc, dtype=int)
        for (r, c), v in list(hbar.items()) + list(wbar.items()):
            gi[r, c] = v
        go = gi.copy()
        for k in range(h // ph + 1):
            for s in (k * ph, -k * ph):
                for (r, c), v in hbar.items():
                    if 0 <= r + s < h:
                        go[r + s, c] = v
        for k in range(w // pw + 1):
            for s in (k * pw, -k * pw):
                for (r, c), v in wbar.items():
                    if 0 <= c + s < w:
                        go[r, c + s] = v
        # Keep only instances whose two runs can be read from the input alone.
        # A run that touches the other line end-on would look one cell longer,
        # which would give the wrong period, so those instances are dropped.
        rd = _read_runs(gi)
        if rd is None or rd[0] != bgc:
            continue
        if rd[1:] != (locjh, locih, locih + ph - 1, lociw, locjw, locjw + pw - 1):
            continue
        return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("no valid instance")


def derive_operations(I, O=None, examples=None):
    """Everything is read from I; O is never looked at.

    The golf entry and the maker read the task the same way. Each line that
    holds a run is rewritten as that run repeated end to end. The period is
    the run's length, and the phase is fixed by where the run starts. The two
    lines are independent of each other. The one real difference is order:
    the golf entry treats each line as a unit and does the rows first, then
    the columns. That order is used here: first the horizontal line, then the
    vertical line.

    On each line the run is copied once with CopyI and pasted tile by tile,
    moving outward from where it sits: first toward the far edge, then back
    toward the near edge. The far edge clips the last tile. The tile cut off
    by the near edge needs its own CopyI of the run's tail. Paste cannot carry
    run cells whose colour is 0, so right after each tile's paste a Color0
    puts in that tile's 0 cells.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    _, cj, v0, v1, ri, h0, h1 = _read_runs(I)

    grid = I.copy()                     # tracks the working grid, to skip no-op ops
    ops, sels = [], []
    state = {"src": None}

    def at(axis, fixed, p):
        return (fixed, p) if axis == 1 else (p, fixed)

    def copy_if_needed(src):
        if state["src"] != src:
            ops.append(28); sels.append(sel_of(src))      # CopyI: the run (or its tail)
            state["src"] = src

    # horizontal line first (rows), then the vertical line (columns)
    for axis, fixed, s0, s1, L in ((1, ri, h0, h1, w), (0, cj, v0, v1, h)):
        n = s1 - s0 + 1
        run = [at(axis, fixed, s0 + i) for i in range(n)]
        vals = [int(I[c]) for c in run]
        fwd = list(range(s0 + n, L, n))           # whole tiles toward the far edge
        bwd = list(range(s0 - n, -1, -n))         # whole tiles toward the near edge
        tiles = [(d, 0, n) for d in fwd + bwd]
        lead = s0 % n                             # tile cut off by the near edge
        if lead > 0:
            tiles.append((0, n - lead, lead))
        for dest, lo, ln in tiles:
            src = run[lo:lo + ln]
            cells = [(at(axis, fixed, dest + i), vals[lo + i])
                     for i in range(ln) if dest + i < L]
            if any(v != 0 and grid[p] != v for p, v in cells):
                copy_if_needed(src)
                ops.append(30); sels.append(sel_of([at(axis, fixed, dest)]))  # Paste
                for p, v in cells:
                    if v != 0:
                        grid[p] = v
            zeros = [p for p, v in cells if v == 0 and grid[p] != 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))  # the 0s a paste cannot carry
                for p in zeros:
                    grid[p] = 0

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == the whole grid, exactly the cells meant
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
