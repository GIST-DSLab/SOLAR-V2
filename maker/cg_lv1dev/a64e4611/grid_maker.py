"""
ARC Task: a64e4611 (RE-ARC) — LLM-generated grid_maker
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

SGN_VARIANTS = [(-1,), (1,), (-1, 1)]


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SGN_VARIANTS):
        ex = [list(v) for v in SGN_VARIANTS]
        ex += [list(random.choice(SGN_VARIANTS)) for _ in range(n_ex - len(SGN_VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [list(v) for v in random.sample(SGN_VARIANTS, n_ex)]
    plan = [{"sgns": s} for s in ex]
    plan.append({"sgns": list(random.choice(ex))})
    return {"bgc": bgc, "noisec": noisec, "instance_plan": plan}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + int(round(diff_lb * (hi - lo)))
    b = lo + int(round(diff_ub * (hi - lo)))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if a > b:
        a, b = b, a
    return random.randint(a, b)


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sgns):
    h = _unif(diff_lb, diff_ub, 18, max(18, min(30, max_h)))
    w = _unif(diff_lb, diff_ub, 18, max(18, min(30, max_w)))
    lb, ub = int(0.4 * h * w), int(0.5 * h * w)
    nbgc = _unif(diff_lb, diff_ub, lb, ub)
    gi = np.full((h, w), noisec, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, nbgc):
        gi[i, j] = bgc
    addn, addb = set(), set()
    for i in range(h - 2):
        for j in range(w - 2):
            blk = gi[i:i + 3, j:j + 3]
            if (blk == bgc).all():
                addn.add((i + random.randint(0, 2), j + random.randint(0, 2)))
            elif (blk == noisec).all():
                addb.add((i + random.randint(0, 2), j + random.randint(0, 2)))
    for (i, j) in addn:
        gi[i, j] = noisec
    for (i, j) in addb:
        gi[i, j] = bgc
    go = gi.copy()
    dim = random.randint(random.randint(3, 8), 8)
    locj = random.randint(3, w - dim - 4)
    spi = random.choice((0, random.randint(3, h // 2)))
    gi[spi:, locj:locj + dim] = bgc
    go[spi:, locj:locj + dim] = bgc
    top = spi + 1 if spi > 0 else spi
    go[top:, locj + 1:locj + dim - 1] = 3

    def arms(sg):
        startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        hh = random.randint(3, min(8, h - startloc - 3))
        for sgn in sg:
            for ii in range(startloc, startloc + hh):
                cs = range(locj, w) if sgn == 1 else range(0, locj + 1)
                for j in cs:
                    gi[ii, j] = bgc
                    if go[ii, j] != 3:
                        go[ii, j] = bgc
        for sgn in sg:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                if sgn == -1:
                    go[ii, 0:locj + dim - 1] = 3
                else:
                    go[ii, locj + 1:w] = 3

    sgns = tuple(sgns)
    arms(sgns)
    if len(sgns) == 1 and _unif(diff_lb, diff_ub, 0, 1) == 1:
        arms((-sgns[0],))
    return gi, go


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sgns=None) -> dict:
    if sgns is None:
        sgns = random.choice(SGN_VARIANTS)
    for _ in range(200):
        gi, go = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sgns)
        # keep only instances whose channel is unambiguous from the input alone
        if np.array_equal(_paint(gi), go):
            return {"input": [list(map(int, r)) for r in gi],
                    "output": [list(map(int, r)) for r in go]}
    raise RuntimeError("could not build an unambiguous instance")


# ── reading of the task ──────────────────────────────────────────────────────
# The background is cut by a channel: a vertical band reaching the bottom edge and
# horizontal arms reaching a side edge.  Walking down the grid row by row, a cell
# of a row is "inside the channel" when it and its row-neighbours above and below
# are background along a background strip that runs to the grid edge (cells off the
# grid count as background).  Each row's inside-channel run is one stroke of 3.

def _bg_color(I):
    h, w = I.shape
    best, bgc = -1, None
    for k in np.unique(I):
        tails = []
        for c in range(w):
            t = 0
            while t < h and I[h - 1 - t, c] == k:
                t += 1
            tails.append(t)
        s = max(min(tails[c:c + 3]) for c in range(w - 2))
        if s > best:
            best, bgc = s, int(k)
    return bgc


def _runs(flags):
    out, i, n = [], 0, len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def _channel_rows(I):
    """dict row -> list of (c0, c1) inside-channel runs, measured from I alone."""
    h, w = I.shape
    bgc = _bg_color(I)
    P = np.full((h + 2, w + 2), True)
    P[1:-1, 1:-1] = (I == bgc)
    mark = np.zeros((h, w), dtype=bool)
    # vertical strip (band): 3-column-clear runs reaching the bottom edge
    for c in range(w):
        ok = [P[r + 1, c:c + 3].all() for r in range(-1, h + 1)]   # padded rows -1..h
        for a, b in _runs(ok):
            if b != h + 1:
                continue
            lo, hi = max(a + 1, 1), min(b - 1, h)
            if hi - lo + 1 >= 8:
                mark[lo - 1:hi, c] = True
    bcols = [c for c in range(w) if mark[:, c].any()]
    bl, br = (min(bcols), max(bcols)) if bcols else (w, -1)
    if bcols:
        # the band is the whole clear column block bl-1..br+1; its top is where
        # every one of those columns has turned background down to the bottom
        tops = []
        for c in range(max(bl - 1, 0), min(br + 1, w - 1) + 1):
            t = h
            while t > 0 and I[t - 1, c] == bgc:
                t -= 1
            tops.append(t)
        spi = max(tops)
        mark[:, :] = False
        mark[(spi + 1 if spi > 0 else 0):, bl:br + 1] = True
    # horizontal strips (arms): 3-row-clear runs touching the left or right edge,
    # whose inner end stops at the band's interior
    for r in range(h):
        ok = [P[r:r + 3, c + 1].all() for c in range(-1, w + 1)]   # padded cols -1..w
        for a, b in _runs(ok):
            if a != 0 and b != w + 1:
                continue
            c0, c1 = max(a, 0), min(b - 2, w - 1)                   # drop run endpoints
            if a != 0:
                c0 = max(c0, bl)
            if b != w + 1:
                c1 = min(c1, br)
            if c1 - c0 + 1 >= 5:
                mark[r, c0:c1 + 1] = True
    rows = {}
    for r in range(h):
        rr = _runs(list(mark[r]))
        if rr:
            rows[r] = rr
    return rows


def _paint(I):
    I = np.asarray(I, dtype=int)
    G = I.copy()
    for r, rr in _channel_rows(I).items():
        for c0, c1 in rr:
            G[r, c0:c1 + 1] = 3
    return G


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # top to bottom: each row's stroke(s) through the channel, painted 3 as a unit;
    # every stroke is a row segment of background measured from I alone
    rows = _channel_rows(I)
    for r in sorted(rows):
        for c0, c1 in rows[r]:
            ops.append(3)
            sels.append(sel_of([(r, c) for c in range(c0, c1 + 1)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task a64e4611"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a64e4611"
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
                                f"for task a64e4611"
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
                    f"Failed to build a complete episode for task a64e4611 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a64e4611-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
