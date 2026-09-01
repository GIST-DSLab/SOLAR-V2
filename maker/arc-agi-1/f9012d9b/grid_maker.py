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


def sample_colors(num_examples=None) -> dict:
    # The rule (periodic-pattern inpainting + crop) does not depend on which
    # colours the wallpaper uses, only on its periodicity. Still, fix the
    # palette once per episode so every instance shares a colour scheme.
    # 0 is hardcoded in the task as the "blanked out" marker, so it is not sampled.
    nc = random.randint(1, 9)
    ccols = random.sample(list(range(1, 10)), nc)
    return {"ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, ccols=None, **kwargs) -> dict:
    if ccols is None:
        ccols = random.sample(list(range(1, 10)), random.randint(1, 9))

    def unifint(lb, ub, bounds):
        a, b = bounds
        return random.randint(int(a + (b - a) * lb), int(a + (b - a) * ub))

    def dmirror(g):
        return [list(r) for r in zip(*g)]

    def vmirror(g):
        return [list(r)[::-1] for r in g]

    def hmirror(g):
        return [list(r) for r in g[::-1]]

    def rot180(g):
        return vmirror(hmirror(g))

    def rot90(g):      # CW
        return dmirror(hmirror(g))

    def rot270(g):     # CCW
        return hmirror(dmirror(g))

    def cmirror(g):
        return dmirror(rot180(g))

    MFS = [(lambda g: [list(r) for r in g], False), (rot90, True),
           (rot180, False), (rot270, True), (dmirror, True),
           (vmirror, False), (hmirror, False), (cmirror, True)]

    mxh = min(int(max_h), 30)
    mxw = min(int(max_w), 30)

    def periods(g):
        h = len(g)
        w = len(g[0])
        pv = h
        for p in range(1, h):
            ok = True
            for r in range(h - p):
                rowa, rowb = g[r], g[r + p]
                for c in range(w):
                    if rowa[c] != 0 and rowb[c] != 0 and rowa[c] != rowb[c]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                pv = p
                break
        ph = w
        for p in range(1, w):
            ok = True
            for row in g:
                for c in range(w - p):
                    if row[c] != 0 and row[c + p] != 0 and row[c] != row[c + p]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                ph = p
                break
        return pv, ph

    def reconstruct(g):
        """Fill the blanked rectangle of g with period-shifted patches of g; return the patch."""
        h, w = len(g), len(g[0])
        zs = [(r, c) for r in range(h) for c in range(w) if g[r][c] == 0]
        if not zs:
            return None
        r0 = min(z[0] for z in zs)
        r1 = max(z[0] for z in zs)
        c0 = min(z[1] for z in zs)
        c1 = max(z[1] for z in zs)
        pv, ph = periods(g)
        work = [list(row) for row in g]
        remaining = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        drs = [k * pv for k in range(-(h // pv + 1), h // pv + 2)]
        dcs = [m * ph for m in range(-(w // ph + 1), w // ph + 2)]
        while remaining:
            best = None
            for dr in drs:
                for dc in dcs:
                    if dr == 0 and dc == 0:
                        continue
                    sr0, sr1 = max(0, r0 - dr), min(h - 1, r1 - dr)
                    sc0, sc1 = max(0, c0 - dc), min(w - 1, c1 - dc)
                    if sr0 > sr1 or sc0 > sc1:
                        continue
                    gain = [(r, c) for (r, c) in remaining
                            if sr0 <= r - dr <= sr1 and sc0 <= c - dc <= sc1
                            and g[r - dr][c - dc] != 0]
                    if not gain:
                        continue
                    key = (-len(gain), abs(dr) + abs(dc), abs(dr), abs(dc))
                    if best is None or key < best[0]:
                        best = (key, dr, dc, gain)
            if best is None:
                return None
            _, dr, dc, gain = best
            for (r, c) in gain:
                work[r][c] = g[r - dr][c - dc]
            remaining -= set(gain)
        return [row[c0:c1 + 1] for row in work[r0:r1 + 1]]

    for _ in range(200):
        mf, transposes = random.choice(MFS)
        # the final mirror/rotation may swap the axes: build within the budget
        # that ends up on each axis after it is applied
        bh, bw = (mxw, mxh) if transposes else (mxh, mxw)
        hp = unifint(diff_lb, diff_ub, (2, max(2, min(10, bh // 3))))
        wp = unifint(diff_lb, diff_ub, (2, max(2, min(10, bw // 3))))
        pat = [[random.choice(ccols) for _ in range(wp)] for _ in range(hp)]
        numhp = unifint(diff_lb, diff_ub, (3, max(3, bh // hp)))
        numwp = unifint(diff_lb, diff_ub, (3, max(3, bw // wp)))
        gi = [[pat[r % hp][c % wp] for c in range(wp * numwp)]
              for r in range(hp * numhp)]
        hcropfac = random.randint(0, hp)
        if hcropfac:
            gi = gi[:len(gi) - hcropfac]
        wcropfac = random.randint(0, wp)
        if wcropfac:
            gi = [row[:len(row) - wcropfac] for row in gi]
        h, w = len(gi), len(gi[0])
        if h > bh or w > bw or h - hp - 1 < 1 or w - wp - 1 < 1:
            continue
        sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
        sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
        loci = min(random.randint(0, h - sgh), h - sgh)
        locj = min(random.randint(0, w - sgw), w - sgw)
        go = [row[locj:locj + sgw] for row in gi[loci:loci + sgh]]
        for r in range(loci, loci + sgh):
            for c in range(locj, locj + sgw):
                gi[r][c] = 0
        gi = [list(r) for r in mf(gi)]
        go = [list(r) for r in mf(go)]
        # keep only instances whose hidden patch is actually recoverable from
        # the visible pattern alone
        if reconstruct(gi) != go:
            continue
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O):
    # Rule: the input is a wallpaper tiling with one rectangle blanked to 0.
    # Restore that rectangle from period-shifted copies of the visible pattern,
    # then crop to it. Everything below is measured from I only; O is never read.
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

    ops, sels = [], []

    # --- fill the hole with patches of pattern taken a whole number of periods
    #     away, greedily, until no blank cell of the hole is left ---
    remaining = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    drs = [k * pv for k in range(-(hi // pv + 1), hi // pv + 2)]
    dcs = [m * ph for m in range(-(wi // ph + 1), wi // ph + 2)]
    while remaining:
        best = None
        for dr in drs:
            for dc in dcs:
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
                key = (-len(gain), abs(dr) + abs(dc), abs(dr), abs(dc))
                if best is None or key < best[0]:
                    best = (key, dr, dc, gain, (sr0, sc0, sr1, sc1))
        if best is None:
            break
        _, dr, dc, gain, (sr0, sc0, sr1, sc1) = best
        # full rectangle: the patch of intact pattern one whole period away
        ops.append(28)
        sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])
        # paste it onto the matching part of the hole (blank cells of the patch
        # write nothing, so an occluded source leaves those cells for later)
        ops.append(30)
        sels.append([sr0 + dr, sc0 + dc, 0, 0])
        remaining -= set(gain)

    # --- the answer is the restored rectangle ---
    # full rectangle: exactly the occluded region
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
