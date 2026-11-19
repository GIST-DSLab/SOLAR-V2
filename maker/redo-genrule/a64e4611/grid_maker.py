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
from random import randint, choice, sample
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # generator samples: bgc, noisec = sample(remove(3, interval(0, 10, 1)), 2)
    # 3 is the fill color and is hardcoded, so it is not a role here.
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def _rearc_verify(gi):
    """Faithful reimplementation of verify_a64e4611: pad with bgc, seed 3 along the
    mid-line of every 3x14 / 14x3 all-bgc block, light up the padding ring, prune
    dead-end 3s (single cells and dominoes walled in on three sides) in all four
    orientations 8 times, then trim the padding away."""
    h = len(gi); w = len(gi[0])
    vals = [v for r in gi for v in r]
    bgc = max(set(vals), key=vals.count)
    H, W = h + 2, w + 2
    g = [[bgc] * W for _ in range(H)]
    for r in range(h):
        row = gi[r]
        for c in range(w):
            g[r + 1][c + 1] = row[c]
    hr = [[0] * W for _ in range(H)]
    for r in range(H):
        run = 0; gr = g[r]; hrr = hr[r]
        for c in range(W):
            run = run + 1 if gr[c] == bgc else 0
            hrr[c] = run
    seeds = set()
    for i in range(H - 2):
        a, b, d = hr[i], hr[i + 1], hr[i + 2]
        for j in range(13, W):
            if a[j] >= 14 and b[j] >= 14 and d[j] >= 14:
                for cc in range(j - 12, j):
                    seeds.add((i + 1, cc))
    vr = [[0] * W for _ in range(H)]
    for c in range(W):
        run = 0
        for r in range(H):
            run = run + 1 if g[r][c] == bgc else 0
            vr[r][c] = run
    for j in range(W - 2):
        for i in range(13, H):
            if vr[i][j] >= 14 and vr[i][j + 1] >= 14 and vr[i][j + 2] >= 14:
                for rr in range(i - 12, i):
                    seeds.add((rr, j + 1))
    for (r, c) in seeds:
        g[r][c] = 3
    for c in range(W):
        g[0][c] = 3; g[H - 1][c] = 3
    for r in range(H):
        g[r][0] = 3; g[r][W - 1] = 3
    for _ in range(8):
        HH = len(g); WW = len(g[0])
        hits = []
        for i in range(HH - 1):
            r0 = g[i]; r1 = g[i + 1]
            for j in range(WW - 3):
                if (r1[j + 1] == 3 and r1[j + 2] == 3 and r0[j + 1] == bgc
                        and r0[j + 2] == bgc and r1[j] == bgc and r1[j + 3] == bgc):
                    hits.append((i, j))
        for (i, j) in hits:
            g[i + 1][j + 1] = bgc; g[i + 1][j + 2] = bgc
        hits = []
        for i in range(HH - 1):
            r0 = g[i]; r1 = g[i + 1]
            for j in range(WW - 2):
                if r1[j + 1] == 3 and r0[j + 1] == bgc and r1[j] == bgc and r1[j + 2] == bgc:
                    hits.append((i, j))
        for (i, j) in hits:
            g[i + 1][j + 1] = bgc
        g = [list(row) for row in zip(*g[::-1])]
    return [row[1:-1] for row in g[1:-1]]


def _params(diff_lb, diff_ub, max_h, max_w):
    """Sample h, w, spi, dim, locj inside the sub-space where the generator's
    intended output equals what the verifier measures:
      spi <= h-13   band tall enough (with padding) to seed a 14-long vertical block
      locj+dim >= 13 / locj <= w-13   left / right arm long enough to seed
      dim >= 5 unless the band starts at the top edge (a 1- or 2-wide interior with a
               free top end is pruned away by the dead-end eroder)"""
    for _ in range(200):
        h = _unifint(diff_lb, diff_ub, (min(18, max_h), max_h))
        w = _unifint(diff_lb, diff_ub, (min(18, max_w), max_w))
        if h < 18 or w < 18:
            return None
        spi_hi = min(h // 2, h - 13)
        spi = 0 if (spi_hi < 3 or choice((0, 1)) == 0) else randint(3, spi_hi)
        dlo = 3 if spi == 0 else 5
        dim = randint(randint(dlo, 8), 8)
        lo = max(3, 13 - dim)
        hi = min(w - 13, min(h, w) - dim - 4)
        if lo > hi:
            continue
        return h, w, spi, dim, randint(lo, hi)
    return None


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec) -> dict:
    for _attempt in range(80):
        p = _params(diff_lb, diff_ub, max_h, max_w)
        if p is None:
            break
        h, w, spi, dim, locj = p
        nbgc = _unifint(diff_lb, diff_ub, (int(0.4 * h * w), int(0.5 * h * w)))
        gi = [[noisec] * w for _ in range(h)]
        inds = [(i, j) for i in range(h) for j in range(w)]
        for (i, j) in sample(inds, nbgc):
            gi[i][j] = bgc
        # break up every 3x3 monochrome patch of the noise field
        addn = set(); addb = set()
        for i in range(h - 2):
            for j in range(w - 2):
                v = gi[i][j]
                if all(gi[i + a][j + b] == v for a in range(3) for b in range(3)):
                    if v == bgc:
                        addn.add((randint(0, 2) + i, randint(0, 2) + j))
                    else:
                        addb.add((randint(0, 2) + i, randint(0, 2) + j))
        for (i, j) in addn:
            gi[i][j] = noisec
        for (i, j) in addb:
            gi[i][j] = bgc
        go = [row[:] for row in gi]
        # vertical band: bgc in input, interior filled with 3 in output
        for j in range(locj, locj + dim):
            for i in range(spi, h):
                gi[i][j] = bgc; go[i][j] = bgc
        for j in range(locj + 1, locj + dim - 1):
            for i in range(spi + 1 if spi > 0 else spi, h):
                go[i][j] = 3
        # horizontal arm(s) shot out of the band, same treatment
        sgns = choice(((-1,), (1,), (-1, 1)))
        startloc = choice((spi, randint(spi + 3, h - 6)))
        plan = [(sgns, startloc, randint(3, min(8, h - startloc - 3)))]
        if len(sgns) == 1 and _unifint(diff_lb, diff_ub, (0, 1)) == 1:
            st2 = choice((spi, randint(spi + 3, h - 6)))
            plan.append(((-sgns[0],), st2, randint(3, min(8, h - st2 - 3))))
        for (ss, st, hh) in plan:
            for sgn in ss:
                for ii in range(st, st + hh):
                    for j in (range(0, locj + 1) if sgn == -1 else range(locj, w)):
                        gi[ii][j] = bgc
                        if go[ii][j] != 3:
                            go[ii][j] = bgc
            for sgn in ss:
                for ii in range(st + 1 if st > 0 else st, st + hh - 1):
                    for j in (range(0, locj + dim - 1) if sgn == -1 else range(locj + 1, w)):
                        go[ii][j] = 3
        vals = [v for r in gi for v in r]
        if max(set(vals), key=vals.count) != bgc:
            continue  # verifier reads mostcolor as the background; keep it true
        if _rearc_verify(gi) != go:
            continue  # noise happened to thicken the structure; resample
        return {"input": gi, "output": go}
    raise ValueError("could not build a verifier-consistent instance")


def derive_operations(I, O):
    """The pipe of background color gets its interior painted 3: the vertical band's
    interior columns, and each horizontal arm's interior rows out to the grid edge.
    Every region is a solid rectangle, measured from I/O, painted whole with Color3."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    # The band always runs to the bottom edge and the arms never reach it,
    # so the last row exposes exactly the band's interior columns.
    band_cols = [c for c in range(wo) if O[ho - 1, c] == 3]
    cb0, cb1 = band_cols[0], band_cols[-1]
    rb0 = min(r for r in range(ho) if O[r, cb0] == 3)   # band interior's top row
    ops.append(3)
    sels.append(sel_of([(r, c) for r in range(rb0, ho) for c in range(cb0, cb1 + 1)]))

    def _runs(rows):
        out, cur = [], []
        for r in rows:
            if cur and r == cur[-1] + 1:
                cur.append(r)
            else:
                if cur:
                    out.append(cur)
                cur = [r]
        if cur:
            out.append(cur)
        return out

    # A left arm is the only thing that can put 3 in column 0; it runs from the far
    # interior column of the band out to the left edge. Mirror image on the right.
    for rows in _runs([r for r in range(ho) if O[r, 0] == 3]):
        ops.append(3)
        sels.append(sel_of([(r, c) for r in rows for c in range(0, cb1 + 1)]))
    for rows in _runs([r for r in range(ho) if O[r, wo - 1] == 3]):
        ops.append(3)
        sels.append(sel_of([(r, c) for r in rows for c in range(cb0, wo)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
