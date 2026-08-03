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
import numpy as np
import random


def sample_colors(num_examples=None) -> dict:
    # Only the color palette of the periodic pattern is random; the rule
    # (fill the 0-hole using the pattern's periodicity, then crop to the hole)
    # is color independent, but we still fix the palette for the whole episode.
    nc = random.randint(1, 9)
    ccols = random.sample(list(range(1, 10)), nc)
    return {"ccols": ccols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, ccols=None) -> dict:
    if ccols is None:
        ccols = random.sample(list(range(1, 10)), random.randint(1, 9))

    m = min(max_h, max_w)          # mf may transpose -> bound both dims by min
    pmax = max(2, min(10, m // 3))

    while True:
        hp = unifint(diff_lb, diff_ub, (2, pmax))
        wp = unifint(diff_lb, diff_ub, (2, pmax))
        srco = canvas(0, (hp, wp))
        inds = asindices(srco)
        obj = {(random.choice(ccols), ij) for ij in inds}
        srco = paint(srco, obj)
        gi = paint(srco, obj)
        numhp = unifint(diff_lb, diff_ub, (3, m // hp))
        numwp = unifint(diff_lb, diff_ub, (3, m // wp))
        for _ in range(numhp - 1):
            gi = vconcat(gi, srco)
        srco = tuple(e for e in gi)
        for _ in range(numwp - 1):
            gi = hconcat(gi, srco)
        hcropfac = random.randint(0, hp)
        for _ in range(hcropfac):
            gi = gi[:-1]
        gi = dmirror(gi)
        wcropfac = random.randint(0, wp)
        for _ in range(wcropfac):
            gi = gi[:-1]
        gi = dmirror(gi)
        h, w = shape(gi)
        if h - hp - 1 < 1 or w - wp - 1 < 1:
            continue
        sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
        sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
        loci = random.randint(0, h - sgh)
        locj = random.randint(0, w - sgw)
        loc = (loci, locj)
        shp = (sgh, sgw)
        obj = {loc, decrement(add(loc, shp))}
        obj = backdrop(obj)
        go = subgrid(obj, gi)
        gi = fill(gi, 0, obj)
        mf = random.choice((
            identity, rot90, rot180, rot270,
            dmirror, vmirror, hmirror, cmirror
        ))
        gi = mf(gi)
        go = mf(go)

        gh, gw = shape(gi)
        if gh > max_h or gw > max_w:
            continue

        I = np.array(gi, dtype=int)
        O = np.array(go, dtype=int)
        if not (I == 0).any():
            continue

        # self-validation: simulate the trajectory derive_operations would emit
        try:
            ops, sels = derive_operations(I, O)
        except Exception:
            continue
        G = I.copy()
        clip = None
        ok = True
        for op, s in zip(ops, sels):
            r, c, hh, ww = s
            if op == 29:
                clip = G[r:r + hh + 1, c:c + ww + 1].copy()
            elif op == 30:
                ch, cw = clip.shape
                tgt = G[r:r + ch, c:c + cw]
                G[r:r + ch, c:c + cw] = np.where(clip != 0, clip, tgt)
            elif op == 33:
                G = G[r:r + hh + 1, c:c + ww + 1].copy()
            elif op == 34:
                pass
            else:
                ok = False
                break
        if not ok or G.shape != O.shape or not np.array_equal(G, O):
            continue

        return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    I: doubly-periodic pattern with one rectangular hole of 0s.
    Rule (measured from I only):
      1. locate the hole rectangle (the 0 cells),
      2. measure the pattern's vertical period ph and horizontal period pw
         from the visible (non-zero) cells of I,
      3. propagate the pattern into the hole by copy/pasting whole bands
         translated by exactly one period,
      4. crop the canvas down to the hole rectangle -> that is the answer.
    O is used only for the final submit bbox.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- 1. hole rectangle -------------------------------------------------
    zr, zc = np.where(I == 0)
    r0, r1 = int(zr.min()), int(zr.max())
    c0, c1 = int(zc.min()), int(zc.max())
    hh = r1 - r0 + 1
    hw = c1 - c0 + 1

    # --- 2. periods measured from I (zeros masked out) ---------------------
    def period(A):
        n = A.shape[0]
        for p in range(1, n):
            good = True
            for r in range(n - p):
                a = A[r]
                b = A[r + p]
                msk = (a != 0) & (b != 0)
                if not np.array_equal(a[msk], b[msk]):
                    good = False
                    break
            if good:
                return p
        return n

    ph = period(I)          # vertical period (row shift)
    pw = period(I.T)        # horizontal period (column shift)

    G = I.copy()
    ops, sels = [], []

    def copy_paste(sr, sc, tr, tc, bh, bw):
        # CopyO: selection is exactly the full source rectangle (bbox is the
        # intended cell set -- a solid block of pattern, no background holes).
        ops.append(29)
        sels.append([sr, sc, bh - 1, bw - 1])
        # Paste: only the top-left corner of the selection matters.
        ops.append(30)
        sels.append([tr, tc, 0, 0])
        G[tr:tr + bh, tc:tc + bw] = G[sr:sr + bh, sc:sc + bw]

    # --- 3. propagate one period at a time into the hole -------------------
    if c1 + pw <= w - 1:
        # pull the pattern in from the right, band by band, right to left
        b = c1
        while b >= c0:
            a = max(c0, b - pw + 1)
            copy_paste(r0, a + pw, r0, a, hh, b - a + 1)
            b = a - 1
    elif c0 - pw >= 0:
        # pull the pattern in from the left, band by band, left to right
        a = c0
        while a <= c1:
            b = min(c1, a + pw - 1)
            copy_paste(r0, a - pw, r0, a, hh, b - a + 1)
            a = b + 1
    elif r1 + ph <= h - 1:
        # pull the pattern up from below, band by band, bottom to top
        b = r1
        while b >= r0:
            a = max(r0, b - ph + 1)
            copy_paste(a + ph, c0, a, c0, b - a + 1, hw)
            b = a - 1
    elif r0 - ph >= 0:
        # pull the pattern down from above, band by band, top to bottom
        a = r0
        while a <= r1:
            b = min(r1, a + ph - 1)
            copy_paste(a - ph, c0, a, c0, b - a + 1, hw)
            a = b + 1
    else:
        # generic fallback: fill period-sized blocks, each from the nearest
        # lattice-equivalent block that is already known (period multiples).
        blocks = [(br, bc)
                  for br in range(r0, r1 + 1, ph)
                  for bc in range(c0, c1 + 1, pw)]
        pending = list(blocks)
        while pending:
            progress = False
            still = []
            for (br, bc) in pending:
                bh = min(ph, r1 - br + 1)
                bw = min(pw, c1 - bc + 1)
                done = False
                for d in range(1, (h // ph + w // pw) + 3):
                    for dk in range(-d, d + 1):
                        dm = d - abs(dk)
                        for sgn in ((1,) if dm == 0 else (1, -1)):
                            sr = br + dk * ph
                            sc = bc + sgn * dm * pw
                            if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                                continue
                            if (G[sr:sr + bh, sc:sc + bw] == 0).any():
                                continue
                            copy_paste(sr, sc, br, bc, bh, bw)
                            done = True
                            break
                        if done:
                            break
                    if done:
                        break
                if done:
                    progress = True
                else:
                    still.append((br, bc))
            pending = still
            if not progress:
                break

    # --- 4. the hole rectangle is the answer -------------------------------
    ops.append(33)
    sels.append([r0, c0, hh - 1, hw - 1])
    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
