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
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # Only the background colour is a real "role" here: the rule (extend the two
    # bars periodically) depends on the *pattern* of the bars, not on which
    # colours the individual bar cells happen to have.  The per-cell bar colours
    # and the crossing colour stay free.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    ph = unifint(diff_lb, diff_ub, (2, 9))
    pw = unifint(diff_lb, diff_ub, (2, 9))
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
def derive_operations(I, O, examples=None):
    """
    Rule: the input holds one vertical bar (a run of ph cells in one column) and
    one horizontal bar (a run of pw cells in one row).  The output stamps the
    vertical bar up/down the whole column with period ph, and the horizontal bar
    left/right along the whole row with period pw.

    Trajectory: CopyI the bar, Paste it at every tile origin.  ARCLE treats 0 as
    "nothing", so bar cells whose colour is 0 do not travel with the paste --
    those exact destination cells are laid in afterwards with one Color0 per
    paste.  When a bar holds no 0, no repair op is emitted at all.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # ---- background ---------------------------------------------------------
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ccount, rcount = Counter(), Counter()
    for r, c in fg:
        ccount[c] += 1
        rcount[r] += 1
    # the bar column holds ph>=2 cells, every other column at most 1
    C = max(ccount.items(), key=lambda kv: kv[1])[0]
    R = max(rcount.items(), key=lambda kv: kv[1])[0]

    colrows = [r for r in range(h) if I[r, C] != bgc]
    rowcols = [c for c in range(w) if I[R, c] != bgc]

    def is_run(x):
        return len(x) >= 2 and x[-1] - x[0] + 1 == len(x)

    # The two bars may touch: a bar cell of one can sit inside the other's line.
    # Enumerate the (at most 3) readings and keep the one that reproduces O.
    cands = []
    if is_run(colrows) and is_run(rowcols):
        cands.append((list(colrows), list(rowcols)))
    if R in colrows:
        v = [r for r in colrows if r != R]
        if is_run(v) and is_run(rowcols):
            cands.append((v, list(rowcols)))
    if C in rowcols:
        hh = [c for c in rowcols if c != C]
        if is_run(colrows) and is_run(hh):
            cands.append((list(colrows), hh))

    def sim(vr, hc):
        G = I.copy()
        ph, rs = len(vr), vr[0]
        pw, cs = len(hc), hc[0]
        for r in range(h):
            G[r, C] = I[rs + (r - rs) % ph, C]
        for c in range(w):
            G[R, c] = I[R, cs + (c - cs) % pw]
        return G

    vrows, hcols = None, None
    for v, hc in cands:
        if np.array_equal(sim(v, hc), O):
            vrows, hcols = v, hc
            break
    if vrows is None:
        vrows, hcols = (cands[0] if cands else (list(colrows), list(rowcols)))

    # ---- tile plan ----------------------------------------------------------
    def cell(axis, fixed, p):
        return (p, fixed) if axis == 0 else (fixed, p)

    def plan_for(axis, idxs, fixed):
        n, start = len(idxs), idxs[0]
        L = h if axis == 0 else w
        vals = [int(I[cell(axis, fixed, start + i)]) for i in range(n)]
        base = start % n
        plans = []
        if base > 0:                                  # partial tile above/left of the first full one
            plans.append((0, list(range(n - base, n))))
        k = base
        while k + n - 1 <= L - 1:
            if k != start:                            # the original bar is already there
                plans.append((k, list(range(n))))
            k += n
        if k <= L - 1:                                # trailing partial tile (paste clips at the edge)
            plans.append((k, list(range(n))))
        return vals, plans, n, start, L

    ops, sels, effs = [], [], []
    cur = I.copy()
    clip_key = None

    for axis, idxs, fixed in ((0, vrows, C), (1, hcols, R)):
        vals, plans, n, start, L = plan_for(axis, idxs, fixed)
        for dest, src in plans:
            key = (axis, tuple((i, vals[j]) for i, j in enumerate(src) if vals[j] != 0))
            changes = any(
                dest + i < L and vals[j] != 0 and cur[cell(axis, fixed, dest + i)] != vals[j]
                for i, j in enumerate(src)
            )
            if changes:
                if clip_key != key:
                    ccells = [cell(axis, fixed, start + j) for j in src]
                    ops.append(28)                       # CopyI: the bar (or its tail) from the input
                    sels.append(sel_of(ccells))
                    effs.append(('copy', ccells))
                    clip_key = key
                dcell = cell(axis, fixed, dest)
                ops.append(30)                           # Paste at this tile origin
                sels.append(sel_of([dcell]))
                effs.append(('paste', dcell))
                for i, j in enumerate(src):
                    if dest + i < L and vals[j] != 0:
                        cur[cell(axis, fixed, dest + i)] = vals[j]
            # cells of this tile whose colour is 0 did not travel with the paste
            zeros = [
                cell(axis, fixed, dest + i)
                for i, j in enumerate(src)
                if dest + i < L and vals[j] == 0 and cur[cell(axis, fixed, dest + i)] != 0
            ]
            if zeros:
                ops.append(0)                            # Color0 on exactly those destinations
                sels.append(sel_of(zeros))
                effs.append(('color0', zeros))
                for rc in zeros:
                    cur[rc] = 0

    # ---- faithful replay (CopyI / Paste / Color0) ---------------------------
    def run_ops(seq):
        G = I.copy()
        clipc = []
        for kind, payload in seq:
            if kind == 'copy':
                r0 = min(r for r, _ in payload)
                c0 = min(c for _, c in payload)
                clipc = [((r - r0, c - c0), int(I[r, c])) for r, c in payload if I[r, c] != 0]
            elif kind == 'paste':
                r0, c0 = payload
                for (dr, dc), v in clipc:
                    r, c = r0 + dr, c0 + dc
                    if 0 <= r < h and 0 <= c < w:
                        G[r, c] = v
            else:
                for r, c in payload:
                    G[r, c] = 0
        return G

    G = run_ops(effs)
    if not np.array_equal(G, O):                        # safety net (should not trigger)
        by_col = {}
        for r in range(h):
            for c in range(w):
                if G[r, c] != O[r, c]:
                    by_col.setdefault(int(O[r, c]), []).append((r, c))
        for v, cells in by_col.items():
            ops.append(v)
            sels.append(sel_of(cells))
            effs.append(('color0' if v == 0 else 'colorX', cells))
        # replay support for the fallback colour ops
        def run_ops2(seq):
            GG = I.copy()
            clipc = []
            for kind, payload in seq:
                if kind == 'copy':
                    r0 = min(r for r, _ in payload)
                    c0 = min(c for _, c in payload)
                    clipc = [((r - r0, c - c0), int(I[r, c])) for r, c in payload if I[r, c] != 0]
                elif kind == 'paste':
                    r0, c0 = payload
                    for (dr, dc), v in clipc:
                        r, c = r0 + dr, c0 + dc
                        if 0 <= r < h and 0 <= c < w:
                            GG[r, c] = v
                else:
                    idx = effs.index((kind, payload))
                    v = ops[idx]
                    for r, c in payload:
                        GG[r, c] = v
            return GG
        run_ops = run_ops2

    # ---- drop any op whose removal still yields O ---------------------------
    i = 0
    while i < len(effs):
        trial = effs[:i] + effs[i + 1:]
        keep_ops = ops[:i] + ops[i + 1:]
        if np.array_equal(run_ops(trial), O):
            del effs[i]
            del ops[i]
            del sels[i]
        else:
            _ = keep_ops
            i += 1

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # bbox == the whole grid, exactly the intended cells
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
