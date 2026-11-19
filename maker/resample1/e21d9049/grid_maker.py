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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level colour role; the bar colours are
    # per-cell random, and the rule depends only on the bars' pattern.
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
        col = random.choice(remcols)
        in_h = locih <= lociw < locih + ph
        in_w = locjw <= locjh < locjw + pw
        # reject ambiguous layouts: the crossing cell belongs to only one bar
        # while touching the end of the other bar (period would be unreadable)
        if in_h and not in_w and (locjw + pw == locjh or locjw - 1 == locjh):
            continue
        if in_w and not in_h and (locih + ph == lociw or locih - 1 == lociw):
            continue
        hbar = {}
        for k in range(ph):
            r = locih + k
            hbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else random.choice(remcols)
        wbar = {}
        for k in range(pw):
            c = locjw + k
            wbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else random.choice(remcols)
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
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # bars hold at most 18 cells on a >=10x10 canvas, so background is the majority
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = I != bgc
    # the bar row / bar column are the only lines holding >= 2 non-background cells
    row = int(np.argmax(fg.sum(axis=1)))
    colj = int(np.argmax(fg.sum(axis=0)))

    def runs(line):
        out, s = [], None
        for i, v in enumerate(list(line) + [False]):
            if v and s is None:
                s = i
            if not v and s is not None:
                out.append((s, i - s))
                s = None
        return out

    vr = max(runs(fg[:, colj]), key=lambda x: x[1])   # vertical bar (start, length)
    hr = max(runs(fg[row, :]), key=lambda x: x[1])    # horizontal bar (start, length)

    G = I.copy()
    ops, sels = [], []

    def tile(vertical):
        s0, L = vr if vertical else hr
        n = hi if vertical else wi
        pat = [int(I[s0 + i, colj]) if vertical else int(I[row, s0 + i]) for i in range(L)]

        def cell(p):
            return (p, colj) if vertical else (row, p)

        tiles = []
        for k in range(-(s0 // L) - 2, n // L + 2):
            if k == 0:
                continue
            d = s0 + k * L
            a = max(0, -d)
            b = min(L, n - d)            # part of the pattern that lands on the grid
            if a >= b:
                continue
            tiles.append((d, a, b))
        # full copies first (one shared clipboard), then the clipped edge copies
        tiles.sort(key=lambda t: (0 if (t[1] == 0 and t[2] == L) else 1, t[0]))
        clip = None
        zero_cells = []
        for d, a, b in tiles:
            changes = [(cell(d + i), pat[i]) for i in range(a, b)
                       if pat[i] != 0 and G[cell(d + i)] != pat[i]]
            zero_cells += [cell(d + i) for i in range(a, b) if pat[i] == 0]
            if not changes:
                continue
            if clip != (a, b):
                src = cell(s0 + a)
                ops.append(28)       # CopyI the bar segment (full rectangle 1xN / Nx1)
                if vertical:
                    sels.append([src[0], src[1], b - a - 1, 0])
                else:
                    sels.append([src[0], src[1], 0, b - a - 1])
                clip = (a, b)
            dst = cell(d + a)
            ops.append(30)
            sels.append([dst[0], dst[1], 0, 0])   # Paste at repeat origin
            for p, v in changes:
                G[p] = v
        # Paste is transparent for colour 0: paint the bar's 0-cells of the repeats explicitly
        zc = [p for p in zero_cells if G[p] != 0]
        if zc:
            ops.append(0)
            sels.append(sel_of(zc))
            for p in zc:
                G[p] = 0

    tile(True)    # repeat the vertical bar along its column with period = its length
    tile(False)   # repeat the horizontal bar along its row with period = its length
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
