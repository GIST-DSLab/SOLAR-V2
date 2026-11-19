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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    return {"bgc": random.choice(range(10))}


def _unifint(lb, ub, lo, hi):
    if hi <= lo:
        return lo
    d = random.uniform(lb, ub)
    return max(lo, min(hi, lo + int(round(d * (hi - lo)))))


def _segments(vals, bgc):
    segs, s = [], None
    for i, v in enumerate(list(vals) + [bgc]):
        if v != bgc and s is None:
            s = i
        elif v == bgc and s is not None:
            segs.append((s, i - 1)); s = None
    return segs


def _read_runs(I, bgc):
    """The column holding the most foreground cells carries the vertical run,
    the row holding the most carries the horizontal one; each run is the
    longest contiguous foreground stretch on its line."""
    fg = (I != bgc)
    cj = int(np.argmax(fg.sum(axis=0)))
    ri = int(np.argmax(fg.sum(axis=1)))
    vr0, vr1 = max(_segments(I[:, cj], bgc), key=lambda s: s[1] - s[0])
    hc0, hc1 = max(_segments(I[ri, :], bgc), key=lambda s: s[1] - s[0])
    return cj, vr0, vr1, ri, hc0, hc1


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    while True:
        h = _unifint(diff_lb, diff_ub, 10, max(10, max_h))
        w = _unifint(diff_lb, diff_ub, 10, max(10, max_w))
        ph = _unifint(diff_lb, diff_ub, 2, 9)
        pw = _unifint(diff_lb, diff_ub, 2, 9)
        hcols = [random.choice(remcols) for _ in range(ph)]
        wcols = [random.choice(remcols) for _ in range(pw)]
        locih = random.randint(0, h - ph)
        locjh = random.randint(0, w - 1)
        locjw = random.randint(0, w - pw)
        lociw = random.randint(0, h - 1)
        col = random.choice(remcols)
        hbar = {}
        for k in range(ph):
            r = locih + k
            hbar[(r, locjh)] = col if abs(r - lociw) % ph == 0 else hcols[k]
        wbar = {}
        for k in range(pw):
            c = locjw + k
            wbar[(lociw, c)] = col if abs(c - locjh) % pw == 0 else wcols[k]
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
        # keep only instances whose two runs can be read back unambiguously
        if Counter(gi.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        if _read_runs(gi, bgc) != (locjh, locih, locih + ph - 1, lociw, locjw, locjw + pw - 1):
            continue
        return {"input": [list(map(int, r)) for r in gi],
                "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O=None, examples=None):
    """Unit of work: ONE COLOUR at a time.

    The vertical run repeats down its column with period = its length, the
    horizontal run along its row likewise.  So every cell of the two lines
    takes the colour of the run cell in its residue class.  For each colour that
    appears in the runs, one Color op paints every line cell whose residue class
    is that colour -- both lines at once.  O is never read; Color carries 0
    like any other colour, so no copy/paste caveats.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    if examples:
        cnt = Counter()
        for ei, _ in examples:
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    cj, vr0, vr1, ri, hc0, hc1 = _read_runs(I, bgc)
    nv, nh = vr1 - vr0 + 1, hc1 - hc0 + 1

    by_colour = {}
    order = []
    for r in range(h):                                  # vertical line
        v = int(I[vr0 + (r - vr0) % nv, cj])
        if v not in by_colour:
            by_colour[v] = []; order.append(v)
        by_colour[v].append((r, cj))
    for c in range(w):                                  # horizontal line
        v = int(I[ri, hc0 + (c - hc0) % nh])
        if v not in by_colour:
            by_colour[v] = []; order.append(v)
        by_colour[v].append((ri, c))

    g = I.copy()
    ops, sels = [], []
    for v in order:
        cells = by_colour[v]
        if all(g[r, c] == v for r, c in cells):         # would change nothing now
            continue
        ops.append(v); sels.append(sel_of(cells))
        for r, c in cells:
            g[r, c] = v
    ops.append(34); sels.append([0, 0, h - 1, w - 1])  # whole grid
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
