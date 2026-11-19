"""
ARC Task: f9012d9b (RE-ARC) — LLM-generated grid_maker
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
from random import sample, choice, randint
import numpy as np
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    # Background/hole colour 0 is hardcoded; the wallpaper colours never matter
    # to the rule (restore the blanked rectangle), so nothing needs fixing.
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **kw) -> dict:
    for _ in range(200):
        ex = _gen_once(diff_lb, diff_ub, max_h, max_w)
        try:
            ops, sels = derive_operations(ex['input'])
        except Exception:
            continue
        # reject degenerate instances where some blank cell has no visible
        # period-translate (the pattern there is unrecoverable from I)
        if _replay(ex['input'], ops, sels) == [list(r) for r in ex['output']]:
            return ex
    raise ValueError('no recoverable instance')


def _gen_once(diff_lb, diff_ub, max_h, max_w):
    cols = interval(1, 10, 1)
    hp = unifint(diff_lb, diff_ub, (2, 10))
    wp = unifint(diff_lb, diff_ub, (2, 10))
    srco = canvas(0, (hp, wp))
    inds = asindices(srco)
    nc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, nc)
    obj = {(choice(ccols), ij) for ij in inds}
    srco = paint(srco, obj)
    gi = paint(srco, obj)
    mx = min(max_h, max_w)
    numhp = unifint(diff_lb, diff_ub, (3, max(3, mx // hp)))
    numwp = unifint(diff_lb, diff_ub, (3, max(3, mx // wp)))
    for k in range(numhp - 1):
        gi = vconcat(gi, srco)
    srco = tuple(e for e in gi)
    for k in range(numwp - 1):
        gi = hconcat(gi, srco)
    hcropfac = randint(0, hp)
    for k in range(hcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    wcropfac = randint(0, wp)
    for k in range(wcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    h, w = shape(gi)
    sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
    sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    loc = (loci, locj)
    shp = (sgh, sgw)
    obj = {loc, decrement(add(loc, shp))}
    obj = backdrop(obj)
    go = subgrid(obj, gi)
    gi = fill(gi, 0, obj)
    mf = choice((
        identity, rot90, rot180, rot270,
        dmirror, vmirror, hmirror, cmirror
    ))
    gi = mf(gi)
    go = mf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading: the hole is a run of blanks inside each of its rows (or columns);
    # every row of a wallpaper repeats with the horizontal period, so the whole
    # band of hole rows is restored by sliding the same band one (or more)
    # period(s) sideways and laying it over the hole. Only if a one-axis slide
    # cannot reach every blank cell do we fall back to diagonal lattice shifts.
    # Everything is measured from I; O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    nz = I != 0

    # --- the occluded rectangle: the only 0 cells in the input ---
    zs = np.argwhere(I == 0)
    r0, c0 = int(zs[:, 0].min()), int(zs[:, 1].min())
    r1, c1 = int(zs[:, 0].max()), int(zs[:, 1].max())
    hh, hw = r1 - r0 + 1, c1 - c0 + 1

    # --- periods of the wallpaper, measured on the visible cells of I ---
    def period(axis):
        n = hi if axis == 0 else wi
        for p in range(1, n):
            if axis == 0:
                a, b = I[:n - p, :], I[p:, :]
                m = nz[:n - p, :] & nz[p:, :]
            else:
                a, b = I[:, :n - p], I[:, p:]
                m = nz[:, :n - p] & nz[:, p:]
            if bool(np.all(a[m] == b[m])):
                return p
        return n

    pv = period(0)
    ph = period(1)
    hole = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    def plan(shifts):
        remaining = set(hole)
        steps = []
        while remaining:
            best = None
            for dr, dc in shifts:
                sr0, sr1 = max(0, r0 - dr), min(hi - 1, r1 - dr)
                sc0, sc1 = max(0, c0 - dc), min(wi - 1, c1 - dc)
                if sr0 > sr1 or sc0 > sc1:
                    continue
                gain = [(r, c) for (r, c) in remaining
                        if sr0 <= r - dr <= sr1 and sc0 <= c - dc <= sc1
                        and I[r - dr, c - dc] != 0]
                if not gain:
                    continue
                key = (-len(gain), abs(dr) + abs(dc), abs(dr), abs(dc), dr, dc)
                if best is None or key < best[0]:
                    best = (key, dr, dc, gain, (sr0, sc0, sr1, sc1))
            if best is None:
                return None
            _, dr, dc, gain, src = best
            steps.append((dr, dc, src))
            remaining -= set(gain)
        return steps

    kh = wi // ph + 1
    kv = hi // pv + 1
    horiz = [(0, m * ph) for m in range(-kh, kh + 1) if m != 0]
    vert = [(k * pv, 0) for k in range(-kv, kv + 1) if k != 0]
    both = [(k * pv, m * ph) for k in range(-kv, kv + 1)
            for m in range(-kh, kh + 1) if (k, m) != (0, 0)]
    steps = plan(horiz)            # slide the band of hole rows along the rows
    if steps is None:
        steps = plan(vert)         # or the band of hole columns along the columns
    if steps is None:
        steps = plan(both)         # diagonal lattice shifts only as a last resort
    if steps is None:
        raise ValueError('hole not recoverable from visible pattern')

    ops, sels = [], []
    for dr, dc, (sr0, sc0, sr1, sc1) in steps:
        # full rectangle: the band of intact pattern a whole period away
        ops.append(28)
        sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
        # lay it over the hole (its blank cells write nothing)
        ops.append(30)
        sels.append([sr0 + dr, sc0 + dc, 0, 0])

    # A copied patch is valid wallpaper wherever it is laid a whole number of
    # periods from its source, so a re-copy is pointless when the clipboard
    # already held, laid at the same spot, restores the same cells. Drop any
    # copy/paste whose removal leaves OUR OWN restoration unchanged (I only).
    crop_sel = [r0, c0, hh - 1, hw - 1]
    target = _replay(I, ops + [33], sels + [crop_sel])
    changed = True
    while changed:
        changed = False
        for k in range(len(ops)):
            o2, s2 = ops[:k] + ops[k + 1:], sels[:k] + sels[k + 1:]
            if o2 and o2[0] == 30:
                continue
            if _replay(I, o2 + [33], s2 + [crop_sel]) == target:
                ops, sels = o2, s2
                changed = True
                break

    # --- the answer is the restored rectangle ---
    # full rectangle: exactly the occluded region
    ops.append(33)
    sels.append(crop_sel)
    ops.append(34)
    sels.append([0, 0, hh - 1, hw - 1])
    return ops, sels


def _replay(I, ops, sels):
    # minimal simulator of CopyI / Paste / CropGrid used by this trajectory
    I = np.asarray(I, dtype=int)
    g = I.copy()
    clip = None
    for op, s in zip(ops, sels):
        if op == 34:
            break
        r, c, h, w = s
        if op == 28:
            clip = I[r:r + h + 1, c:c + w + 1].copy()
        elif op == 30:
            if clip is None:
                continue
            H, W = clip.shape
            for i in range(H):
                for j in range(W):
                    if clip[i, j] and r + i < g.shape[0] and c + j < g.shape[1]:
                        g[r + i, c + j] = clip[i, j]
        elif op == 33:
            g = g[r:r + h + 1, c:c + w + 1].copy()
    return g.tolist()


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
                        f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f9012d9b"
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
                                f"for task f9012d9b"
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
                    f"Failed to build a complete episode for task f9012d9b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f9012d9b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
