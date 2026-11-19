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


# ----------------------------------------------------------------------------
# 1. episode-level colors
# ----------------------------------------------------------------------------
# The rule ("repeat each bar's colour pattern periodically along its own row /
# column") does not depend on WHICH colours the bar cells carry -- only on the
# pattern being present.  So only the background needs to be fixed per episode.
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator (RE-ARC generator with max_h/max_w bounds and injected bgc)
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    ph = unifint(diff_lb, diff_ub, (2, 9))
    pw = unifint(diff_lb, diff_ub, (2, 9))
    if bgc is None:
        bgc = choice(cols)
    remcols = remove(bgc, cols)
    hbar = frozenset({(choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(choice(remcols), (0, k)) for k in range(pw)})
    locih = randint(0, h - ph)
    locjh = randint(0, w - 1)
    loch = (locih, locjh)
    locjw = randint(0, w - pw)
    lociw = randint(0, h - 1)
    locw = (lociw, locjw)
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, loch)
    wbar = shift(wbar, locw)
    col = choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    go = paint(canv, hbar | wbar)
    for k in range(h // ph + 1):
        go = paint(go, shift(hbar, (k * ph, 0)))
        go = paint(go, shift(hbar, (-k * ph, 0)))
    for k in range(w // pw + 1):
        go = paint(go, shift(wbar, (0, k * pw)))
        go = paint(go, shift(wbar, (0, -k * pw)))
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def _longest_run(vals, bgc):
    """(start, length) of the longest contiguous non-background run."""
    best_s, best_l = 0, 0
    i, n = 0, len(vals)
    while i < n:
        if vals[i] != bgc:
            j = i
            while j < n and vals[j] != bgc:
                j += 1
            if j - i > best_l:
                best_s, best_l = i, j - i
            i = j
        else:
            i += 1
    return best_s, best_l


def _largest_period(seq, maxp):
    """Largest p <= maxp that is a true period of the whole line."""
    n = len(seq)
    best = None
    for p in range(1, min(maxp, n) + 1):
        if all(seq[i] == seq[i + p] for i in range(n - p)):
            best = p
    return best if best is not None else max(1, min(maxp, n))


def _stamp(cur, O, ops, sels, cells):
    """Paint one repetition of the bar pattern: one Color op per colour of the
    stamp, skipping cells that already hold the colour (those ops would be
    no-ops)."""
    todo = [(r, c) for (r, c) in cells if cur[r, c] != O[r, c]]
    by_col = {}
    for (r, c) in todo:
        by_col.setdefault(int(O[r, c]), []).append((r, c))
    for v in sorted(by_col, key=lambda k: by_col[k][0]):
        pts = by_col[v]
        ops.append(v)
        sels.append(sel_of(pts))
        for (r, c) in pts:
            cur[r, c] = v


def derive_operations(I, O):
    # Rule: the grid holds one horizontal bar (a short colour sequence in a row)
    # and one vertical bar (a colour sequence in a column).  Each bar's pattern
    # is repeated periodically outward along its own line until the whole row /
    # column is filled.  Bar cells may be colour 0, so Copy/Paste (which treats
    # 0 as transparent) cannot stamp them -- the repetitions are painted with
    # explicit Color ops, one stamp (one period) at a time, growing outward
    # from the original bar.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the horizontal bar's row and the vertical bar's column are the row / column
    # carrying the most non-background cells (>=2 vs at most 1 elsewhere)
    rowc = [int(np.sum(I[r] != bgc)) for r in range(h)]
    colc = [int(np.sum(I[:, c] != bgc)) for c in range(w)]
    r0 = max(range(h), key=lambda r: rowc[r])
    c0 = max(range(w), key=lambda c: colc[c])

    cur = I.copy()

    # ---- horizontal bar: repeat its pattern along row r0 -------------------
    a, L = _longest_run(list(I[r0]), bgc)
    if L > 0:
        p = _largest_period([int(x) for x in O[r0]], L)
        starts = []
        s = a + p
        while s < w:                      # outward to the right
            starts.append(s)
            s += p
        s = a - p
        while s + p > 0:                  # outward to the left
            starts.append(s)
            s -= p
        for s in starts:
            _stamp(cur, O, ops, sels,
                   [(r0, c) for c in range(s, s + p) if 0 <= c < w])

    # ---- vertical bar: repeat its pattern along column c0 ------------------
    u, M = _longest_run([int(x) for x in I[:, c0]], bgc)
    if M > 0:
        q = _largest_period([int(x) for x in O[:, c0]], M)
        starts = []
        s = u + q
        while s < h:                      # outward downward
            starts.append(s)
            s += q
        s = u - q
        while s + q > 0:                  # outward upward
            starts.append(s)
            s -= q
        for s in starts:
            _stamp(cur, O, ops, sels,
                   [(r, c0) for r in range(s, s + q) if 0 <= r < h])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
