"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- colors
def sample_colors(num_examples=None) -> dict:
    # only bgc is a real color role here: the rule (complete the 4-fold
    # rotational symmetry about the intact core) is color independent.
    cols = list(range(10))
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ---------------------------------------------------------------- generator
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] < d // 2)
        cp = {(d // 2 - 1, d // 2 - 1), (d // 2, d // 2 - 1), (d // 2 - 1, d // 2), (d // 2, d // 2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] <= d // 2)
        cp = {(d // 2, d // 2)} | ineighbors((d // 2, d // 2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d - k - 1, d - k - 1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- ops
def _plan_rotations(I, O, fg, r0, c0, k):
    """Rule measured from I: the largest 4-fold-symmetric k x k window at
    (r0,c0) is the intact core; its centre is the rotation centre.
    Trajectory: rotate the whole symmetric square 90 deg CW in place, then
    stamp the original input pattern back on top; repeat 3 times, which
    accumulates I | rot90(I) | rot180(I) | rot270(I) about that centre."""
    hi, wi = I.shape
    CR = 2 * r0 + k - 1          # doubled coords of the core centre
    CC = 2 * c0 + k - 1
    t = 0
    for (r, c) in fg:
        t = max(t, abs(2 * r - CR), abs(2 * c - CC))
    if (t - (k - 1)) % 2 != 0:
        return None
    R = (CR - t) // 2
    C = (CC - t) // 2
    L = t + 1                                  # smallest rotation-invariant
    if L < 2 or R < 0 or C < 0 or R + L > hi or C + L > wi:   # square holding
        return None                                           # all of I's fg
    # full rectangle on purpose: the rotation carries the whole region,
    # background included, about its centre.
    square = [(r, c) for r in range(R, R + L) for c in range(C, C + L)]

    A = {(r, c): int(I[r, c]) for (r, c) in fg}     # the original object
    colors = sorted(set(A.values()))
    cur = I.copy()
    ops, sels = [], []
    for _ in range(3):
        ops.append(25)                               # Rotate270 = 90 deg CW
        sels.append(sel_of(square))
        cur[R:R + L, C:C + L] = np.rot90(cur[R:R + L, C:C + L], 3)
        for col in colors:                           # stamp the original back
            cells = [p for p in sorted(A) if A[p] == col and cur[p[0], p[1]] != col]
            if cells:
                ops.append(col)
                sels.append(sel_of(cells))
                for (r, c) in cells:
                    cur[r, c] = col
    if not np.array_equal(cur, O):
        return None
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
    return ops, sels


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    if np.array_equal(I, O):
        return [34], [[0, 0, hi - 1, wi - 1]]

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    fg = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    if fg:
        minr = min(r for r, _ in fg); maxr = max(r for r, _ in fg)
        minc = min(c for _, c in fg); maxc = max(c for _, c in fg)
        fcr = minr + (maxr - minr + 1) // 2
        fcc = minc + (maxc - minc + 1) // 2

        # find every fully 4-fold rotationally symmetric square window inside
        # the foreground bbox; the largest (tie-break: closest to fg centre)
        # is the intact core that fixes the symmetry centre.
        cands = []
        for k in range(7, 1, -1):
            for r0 in range(minr, maxr + 2 - k):
                for c0 in range(minc, maxc + 2 - k):
                    win = I[r0:r0 + k, c0:c0 + k]
                    if win.shape != (k, k):
                        continue
                    if int(np.count_nonzero(win == bgc)) >= k * k - 1:
                        continue
                    if not (np.array_equal(win, np.rot90(win, 1))
                            and np.array_equal(win, np.rot90(win, 2))
                            and np.array_equal(win, np.rot90(win, 3))):
                        continue
                    dist = abs(r0 + k // 2 - fcr) + abs(c0 + k // 2 - fcc)
                    cands.append((-k, dist, r0, c0, k))
        cands.append((-1, 0, fcr, fcc, 1))          # degenerate fallback core
        cands.sort()
        for (_nk, _d, r0, c0, k) in cands:
            plan = _plan_rotations(I, O, fg, r0, c0, k)
            if plan is not None:
                return plan

    # defensive fallback (no symmetry centre reproduces O): paint the missing
    # cells grouped by colour.
    ops, sels = [], []
    by_color = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != O[r, c]:
                by_color.setdefault(int(O[r, c]), []).append((r, c))
    for col in sorted(by_color):
        ops.append(col)
        sels.append(sel_of(by_color[col]))
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
