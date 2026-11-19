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
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The rule (restore the blanked part of a periodic wallpaper) never depends
    # on which colours the tile uses, and 0 is reserved for the blank hole.
    # The tile palette is sampled freely per instance inside generate().
    return {}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, **color_kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(max(a, lo), min(b, hi))

    def tr(g):
        return [list(r) for r in zip(*g)]

    transforms = ["identity", "rot90", "rot180", "rot270",
                  "dmirror", "vmirror", "hmirror", "cmirror"]
    mf = random.choice(transforms)
    swaps = mf in ("rot90", "rot270", "dmirror", "cmirror")
    LH, LW = (max_w, max_h) if swaps else (max_h, max_w)

    cols = list(range(1, 10))
    hp = unifint(diff_lb, diff_ub, (2, max(2, min(10, LH // 3))))
    wp = unifint(diff_lb, diff_ub, (2, max(2, min(10, LW // 3))))
    nc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = random.sample(cols, nc)
    tile = [[random.choice(ccols) for _ in range(wp)] for _ in range(hp)]
    numhp = unifint(diff_lb, diff_ub, (3, max(3, LH // hp)))
    numwp = unifint(diff_lb, diff_ub, (3, max(3, LW // wp)))
    H, W = hp * numhp, wp * numwp
    gi = [[tile[r % hp][c % wp] for c in range(W)] for r in range(H)]
    hcrop = random.randint(0, hp)
    wcrop = random.randint(0, wp)
    gi = [row[:W - wcrop] for row in gi[:H - hcrop]]
    h, w = len(gi), len(gi[0])
    sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
    sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
    loci = random.randint(0, h - sgh)
    locj = random.randint(0, w - sgw)
    go = [row[locj:locj + sgw] for row in gi[loci:loci + sgh]]
    for r in range(loci, loci + sgh):
        for c in range(locj, locj + sgw):
            gi[r][c] = 0

    def apply(g):
        if mf == "identity":
            return [list(r) for r in g]
        if mf == "rot90":
            return [list(r) for r in zip(*g[::-1])]
        if mf == "rot180":
            return [list(r[::-1]) for r in g[::-1]]
        if mf == "rot270":
            return [list(r) for r in zip(*g)][::-1]
        if mf == "dmirror":
            return tr(g)
        if mf == "vmirror":
            return [list(r[::-1]) for r in g]
        if mf == "hmirror":
            return [list(r) for r in g[::-1]]
        if mf == "cmirror":
            return [list(r) for r in tr([list(x[::-1]) for x in g[::-1]])]
        return g

    return {"input": apply(gi), "output": apply(go)}


def derive_operations(I, O=None, examples=None):
    # Reading: the input is a wallpaper with one rectangle blanked to 0.
    # Each blank cell equals the cell one (or a few) horizontal periods away IN
    # THE SAME ROW. So the unit of repair is the band of hole rows: it is
    # restored by copying a strip of those same rows from a whole number of
    # horizontal periods to the left/right and pasting it over the hole.
    # Vertical shifts are used only for cells that no same-row source can
    # reach. Then the restored rectangle is cropped out. O is never read.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    nz = I != 0

    zs = np.argwhere(I == 0)
    r0, c0 = int(zs[:, 0].min()), int(zs[:, 1].min())
    r1, c1 = int(zs[:, 0].max()), int(zs[:, 1].max())
    hh, hw = r1 - r0 + 1, c1 - c0 + 1

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

    ops, sels = [], []
    remaining = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    known = nz.copy()  # cells of the working grid that already hold pattern

    dcs = [m * ph for m in range(-(wi // ph + 1), wi // ph + 2)]
    drs = [k * pv for k in range(-(hi // pv + 1), hi // pv + 2)]

    def best_shift(cands):
        best = None
        for dr, dc in cands:
            if dr == 0 and dc == 0:
                continue
            sr0, sr1 = max(0, r0 - dr), min(hi - 1, r1 - dr)
            sc0, sc1 = max(0, c0 - dc), min(wi - 1, c1 - dc)
            if sr0 > sr1 or sc0 > sc1:
                continue
            gain = [(r, c) for (r, c) in remaining
                    if sr0 <= r - dr <= sr1 and sc0 <= c - dc <= sc1
                    and I[r - dr, c - dc] != 0]
            if not gain:
                continue
            key = (abs(dr) + abs(dc), -len(gain), abs(dr), abs(dc))
            if best is None or key < best[0]:
                best = (key, dr, dc, gain, (sr0, sc0, sr1, sc1))
        return best

    stages = [
        [(0, dc) for dc in dcs],                       # same-row sources first
        [(dr, 0) for dr in drs],                       # then same-column
        [(dr, dc) for dr in drs for dc in dcs],        # finally diagonal
    ]
    for cands in stages:
        while remaining:
            best = best_shift(cands)
            if best is None:
                break
            _, dr, dc, gain, (sr0, sc0, sr1, sc1) = best
            # full rectangle: the strip of the hole's rows (or columns) lying
            # a whole number of periods away, copied from the input
            ops.append(28)
            sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
            # paste lands on the matching part of the hole; blank source cells
            # write nothing and are left for a later strip
            ops.append(30)
            sels.append([sr0 + dr, sc0 + dc, 0, 0])
            remaining -= set(gain)

    # full rectangle: exactly the (now restored) blanked region
    ops.append(33)
    sels.append([r0, c0, hh - 1, hw - 1])
    ops.append(34)
    sels.append([0, 0, hh - 1, hw - 1])
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
