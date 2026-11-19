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
# reading the two runs out of an input grid (used by generate + derive_operations)
# ----------------------------------------------------------------------------
def _read_runs(I):
    """Locate the vertical run and the horizontal run in an input grid.

    Returns (col, r0, r1, row, c0, c1): the vertical run occupies rows r0..r1 of
    column `col`, the horizontal run occupies cols c0..c1 of row `row`.

    The vertical run's column is the only column carrying two or more marked
    cells (every other column carries at most one cell of the horizontal run),
    and symmetrically for the horizontal run's row.  Where the two runs meet,
    the shared cell may belong to either run; it is read as the horizontal run's
    cell only when the vertical line's two ends carry the same colour, which is
    what the crossing demands if the vertical run really stops one cell short
    (and vice versa).
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(h) for c in range(w) if I[r, c] != bgc]
    ccount, rcount = Counter(), Counter()
    for r, c in fg:
        ccount[c] += 1
        rcount[r] += 1
    C = max(ccount.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    R = max(rcount.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def longest_run(ps):
        runs, s = [], ps[0]
        for i in range(1, len(ps)):
            if ps[i] != ps[i - 1] + 1:
                runs.append((s, ps[i - 1]))
                s = ps[i]
        runs.append((s, ps[-1]))
        return max(runs, key=lambda ab: ab[1] - ab[0])

    r0, r1 = longest_run(sorted(r for r, c in fg if c == C))
    c0, c1 = longest_run(sorted(c for r, c in fg if r == R))

    if I[R, C] != bgc and r0 <= R <= r1 and c0 <= C <= c1:
        L, M = r1 - r0 + 1, c1 - c0 + 1
        vdrop = R in (r0, r1) and L >= 3 and I[r0, C] == I[r1, C]
        hdrop = C in (c0, c1) and M >= 3 and I[R, c0] == I[R, c1]
        if vdrop and hdrop:                      # only one of them can be true
            if (h - L + 2) * (w - M + 1) <= (h - L + 1) * (w - M + 2):
                hdrop = False
            else:
                vdrop = False
        if vdrop:
            r0, r1 = (r0 + 1, r1) if R == r0 else (r0, r1 - 1)
        elif hdrop:
            c0, c1 = (c0 + 1, c1) if C == c0 else (c0, c1 - 1)
    return C, r0, r1, R, c0, c1


def _tiled(I):
    """The grid the rule asks for: each run repeated along its own line."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    C, r0, r1, R, c0, c1 = _read_runs(I)
    P = I.copy()
    n = r1 - r0 + 1
    for r in range(h):
        P[r, C] = I[r0 + (r - r0) % n, C]
    m = c1 - c0 + 1
    for c in range(w):
        P[R, c] = I[R, c0 + (c - c0) % m]
    return P


# ----------------------------------------------------------------------------
# 1. sample_colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # The background is the only real colour role: the rule ("repeat each run
    # along its own line") depends on the pattern of the runs, not on which
    # colours their cells happen to carry, so the run colours stay free.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generate
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    remcols = remove(bgc, cols)

    def build():
        h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        ph = unifint(diff_lb, diff_ub, (2, min(9, h)))
        pw = unifint(diff_lb, diff_ub, (2, min(9, w)))
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
        return gi, go

    gi, go = build()
    # Where one run ends right against the other run's line, the input can read
    # as two different pairs of runs -- such a grid does not carry its own
    # answer. Keep drawing until the grid says unambiguously which runs it holds.
    for _ in range(64):
        if np.array_equal(_tiled(np.array(gi, dtype=int)), np.array(go, dtype=int)):
            break
        gi, go = build()
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. derive_operations
# ----------------------------------------------------------------------------
def derive_operations(I, O=None, examples=None):
    """Everything is read from the INPUT (O is never inspected).

    The input carries one short vertical run of colours in some column and one
    short horizontal run in some row.  Each run repeats along its own line, in
    both directions, out to the edges of the grid: the vertical run tiles up and
    down its column with a period equal to its own length, the horizontal run
    tiles left and right along its row.

    Trajectory: CopyI the run once, then Paste it at each tile origin down/along
    the line.  ARCLE reads 0 as "nothing there", so a run cell whose colour is 0
    does not travel with the paste -- exactly those destination cells are laid
    in with a Color0 right after the paste that should have carried them (a run
    holding no 0 gets no such op at all).
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    MAXD = 30                                   # ARCLE's padded canvas

    cj, vr0, vr1, ri, hc0, hc1 = _read_runs(I)
    pred = _tiled(I)                            # the grid this rule asks for

    # ---- a faithful replay of the ops used here (CopyI / Paste / Color) -----
    def blank():
        g = np.zeros((MAXD, MAXD), dtype=int)
        g[:h, :w] = I
        return {"grid": g, "clip": np.zeros((MAXD, MAXD), dtype=int), "cd": (0, 0)}

    def step(st, op, cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if op == 28:                                     # CopyI
            st["clip"][:, :] = 0
            ch, cw = r1 - r0 + 1, c1 - c0 + 1
            st["cd"] = (ch, cw)
            sub = I[r0:r1 + 1, c0:c1 + 1]
            msk = np.zeros((ch, cw), dtype=bool)
            for r, c in cells:
                msk[r - r0, c - c0] = True
            np.copyto(st["clip"][:ch, :cw], sub, where=np.logical_and(sub > 0, msk))
        elif op == 30:                                   # Paste at the selection's top-left
            ch, cw = st["cd"]
            if ch == 0 or cw == 0:
                return
            er, ec = min(r0 + ch, MAXD), min(c0 + cw, MAXD)
            patch = st["clip"][:ch, :cw][:er - r0, :ec - c0]
            np.copyto(st["grid"][r0:er, c0:ec], patch, where=(patch > 0))
        else:                                            # Color<op>
            for r, c in cells:
                st["grid"][r, c] = op

    def replay(acts):
        st = blank()
        for op, cells in acts:
            step(st, op, cells)
        return st["grid"][:h, :w]

    # ---- lay the tiles -----------------------------------------------------
    def at(axis, fixed, p):
        return (p, fixed) if axis == 0 else (fixed, p)

    acts = []
    st = blank()
    clip_src = None

    for axis, fixed, s0, s1, L in ((0, cj, vr0, vr1, h), (1, ri, hc0, hc1, w)):
        n = s1 - s0 + 1
        vals = [int(I[at(axis, fixed, s0 + i)]) for i in range(n)]
        base = s0 % n
        tiles = []
        if base > 0:
            # the repetition cut off by the top / left edge shows the run's tail
            tiles.append((0, n - base, base))
        k = base
        while k <= L - 1:
            # whole copies of the run, down/along the line; the last one hangs
            # over the far edge and ARCLE clips it there
            tiles.append((k, 0, n))
            k += n

        for dest, lo, ln in tiles:
            src = [at(axis, fixed, s0 + lo + i) for i in range(ln)]
            dcell = at(axis, fixed, dest)
            before = st["grid"][:h, :w].copy()
            trial = {"grid": st["grid"].copy(), "clip": st["clip"].copy(), "cd": st["cd"]}
            if clip_src != src:
                step(trial, 28, src)
            step(trial, 30, [dcell])
            if not np.array_equal(trial["grid"][:h, :w], before):
                if clip_src != src:
                    acts.append((28, src))               # CopyI: the run itself
                    step(st, 28, src)
                    clip_src = src
                acts.append((30, [dcell]))               # Paste at this tile's origin
                step(st, 30, [dcell])
            zeros = [at(axis, fixed, dest + i) for i in range(ln)
                     if dest + i <= L - 1 and vals[lo + i] == 0
                     and st["grid"][at(axis, fixed, dest + i)] != 0]
            if zeros:
                acts.append((0, zeros))                  # the 0s a paste cannot carry
                step(st, 0, zeros)

    # ---- anything a copy could not carry, painted from the read-off rule ----
    got = st["grid"][:h, :w]
    if not np.array_equal(got, pred):
        rest = {}
        for r in range(h):
            for c in range(w):
                if got[r, c] != pred[r, c]:
                    rest.setdefault(int(pred[r, c]), []).append((r, c))
        for v, cells in sorted(rest.items()):
            acts.append((v, cells))
            step(st, v, cells)

    # ---- drop any action the finished grid does not need --------------------
    target = replay(acts)
    dropped = True
    while dropped:
        dropped = False
        for i in range(len(acts)):
            if np.array_equal(replay(acts[:i] + acts[i + 1:]), target):
                del acts[i]
                dropped = True
                break

    ops = [op for op, _ in acts]
    sels = [sel_of(cells) for _, cells in acts]
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
