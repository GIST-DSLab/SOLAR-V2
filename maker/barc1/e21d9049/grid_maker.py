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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return max(a, min(b, int(round(random.uniform(lo, hi)))))


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; run colours are free per instance
    # (the rule is "repeat the run", independent of which colours it holds).
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    for _ in range(1000):
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        ph = _unifint(diff_lb, diff_ub, (2, 9))
        pw = _unifint(diff_lb, diff_ub, (2, 9))
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        vrows = range(locih, locih + ph)
        hcols = range(locjw, locjw + pw)
        # reject layouts where the other bar's stray cell butts against a run end
        # (the run's length -- its period -- would then be unreadable)
        if locjh in hcols and lociw not in vrows and lociw in (locih - 1, locih + ph):
            continue
        if lociw in vrows and locjh not in hcols and locjh in (locjw - 1, locjw + pw):
            continue
        break
    col = random.choice(remcols)
    hbar = {}
    for k in range(ph):
        r = locih + k
        hbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else random.choice(remcols)
    wbar = {}
    for k in range(pw):
        c = locjw + k
        wbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else random.choice(remcols)
    gi = [[bgc] * w for _ in range(h)]
    for d in (hbar, wbar):
        for (r, c), v in d.items():
            gi[r][c] = v
    go = [row[:] for row in gi]
    for k in range(-(h // ph + 1), h // ph + 2):
        for (r, c), v in hbar.items():
            rr = r + k * ph
            if 0 <= rr < h:
                go[rr][c] = v
    for k in range(-(w // pw + 1), w // pw + 2):
        for (r, c), v in wbar.items():
            cc = c + k * pw
            if 0 <= cc < w:
                go[r][cc] = v
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Everything is read from I (O is never inspected).

    I holds one multicoloured vertical run (column cj) and one horizontal run
    (row ri).  Following the reference reading: first expand the horizontal
    run along its row -- rightward, then leftward -- then the vertical run along
    its column -- downward, then upward.  Tiles are aligned to the run's own
    start, period = run length.  Each tile = CopyI of the run + Paste at the
    tile origin; a tile cut off by the left/top edge pastes only the run's
    visible tail at index 0.  Run cells coloured 0 cannot travel with a paste,
    so they are laid with Color0 on the cells that tile lands on.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]   # canvas colour (bars are tiny)
    nb = I != bgc
    cj = int(np.argmax(nb.sum(axis=0)))   # only the vertical run puts >=2 cells in a column
    ri = int(np.argmax(nb.sum(axis=1)))   # only the horizontal run puts >=2 cells in a row

    def longest_seg(m):
        best, s = None, None
        for i in range(len(m) + 1):
            if i < len(m) and m[i]:
                if s is None:
                    s = i
            elif s is not None:
                if best is None or i - s > best[1] - best[0] + 1:
                    best = (s, i - 1)
                s = None
        return best

    vr0, vr1 = longest_seg(nb[:, cj])
    hc0, hc1 = longest_seg(nb[ri, :])

    G = I.copy()
    ops, sels = [], []
    clip_key = None
    for axis in ("h", "v"):
        if axis == "h":
            L, s0, s1 = w, hc0, hc1
            pos = lambda p: (ri, p)
        else:
            L, s0, s1 = h, vr0, vr1
            pos = lambda p: (p, cj)
        n = s1 - s0 + 1
        vals = [int(I[pos(s0 + i)]) for i in range(n)]
        tiles = []
        k = s0 + n                      # tiles toward the right / bottom
        while k < L:
            tiles.append((k, 0, n))
            k += n
        k = s0 - n                      # tiles toward the left / top
        while k > -n:
            if k >= 0:
                tiles.append((k, 0, n))
            else:                       # cut by the edge: its visible tail only
                tiles.append((0, -k, n + k))
            k -= n
        for dest, lo, ln in tiles:
            src = [pos(s0 + lo + i) for i in range(ln)]
            land = [(dest + i, vals[lo + i]) for i in range(ln) if dest + i < L]
            if any(v != 0 and G[pos(p)] != v for p, v in land):
                key = (axis, lo, ln)
                if clip_key != key:
                    ops.append(28); sels.append(sel_of(src))          # CopyI the run (or its tail)
                    clip_key = key
                ops.append(30); sels.append(sel_of([pos(dest)]))      # Paste at the tile origin
                for p, v in land:
                    if v != 0:
                        G[pos(p)] = v
            zeros = [pos(p) for p, v in land if v == 0 and G[pos(p)] != 0]
            if zeros:
                ops.append(0); sels.append(sel_of(zeros))             # the 0s a paste can't carry
                for c in zeros:
                    G[c] = 0
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
