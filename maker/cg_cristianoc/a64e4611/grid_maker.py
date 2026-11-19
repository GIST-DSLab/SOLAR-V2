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

# Reading shared with the code-golf entry: the 3-cells are the *deep interior* of the
# background-coloured corridor system -- background cells whose whole 8-neighbourhood is
# background too (off-grid counts as open). Units, in order: first the main vertical
# corridor (the largest open rectangle anchored at the bottom edge), then each side arm,
# taken as its own rectangle running from the corridor's side wall out to the grid edge,
# so it does not overlap the corridor. Larger arms go first. Every rectangle is measured
# from I alone; the band colour is an episode-level role read from the demonstrations.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def _unifint(lb, ub, rng):
    a, b = rng
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round(d * (b - a) + random.uniform(-0.5, 0.5)))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    if bgc is None or noisec is None:
        bgc, noisec = random.sample([c for c in range(10) if c != 3], 2)
    last = None
    for _ in range(50):
        last = _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec)
        # reject instances whose noise accidentally widens/extends the corridor so the
        # input no longer determines the output
        gi = np.asarray(last["input"]); go = np.asarray(last["output"])
        pred = gi.copy()
        for cells in _detect(gi, bgc):
            for r, c in cells:
                pred[r, c] = 3
        if (pred == go).all():
            return last
    return last


def _generate_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec):
    h = _unifint(diff_lb, diff_ub, (min(18, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(18, max_w), max_w))
    nbgc = _unifint(diff_lb, diff_ub, (int(0.4 * h * w), int(0.5 * h * w)))
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
    locj = random.randint(3, min(h, w) - dim - 4)
    spi = random.choice((0, random.randint(3, h // 2)))
    gi[spi:, locj:locj + dim] = bgc
    go[spi:, locj:locj + dim] = bgc
    go[(spi + 1 if spi > 0 else spi):, locj + 1:locj + dim - 1] = 3

    def arm(sgn):
        startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        hh = random.randint(3, min(8, h - startloc - 3))
        for ii in range(startloc, startloc + hh):
            cs = range(0, locj + 1) if sgn == -1 else range(locj, w)
            for j in cs:
                gi[ii, j] = bgc
                if go[ii, j] != 3:
                    go[ii, j] = bgc
        for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
            cs = range(0, locj + dim - 1) if sgn == -1 else range(locj + 1, w)
            for j in cs:
                go[ii, j] = 3

    sgns = random.choice(((-1,), (1,), (-1, 1)))
    for s in sgns:
        arm(s)
    if len(sgns) == 1 and _unifint(diff_lb, diff_ub, (0, 1)) == 1:
        arm(-sgns[0])
    return {"input": gi.tolist(), "output": go.tolist()}


def _band_colour(I, examples):
    # the band colour is a fixed episode role: read it from the demonstrations
    # (the input colour under the cells the rule turned to 3).
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int); eo = np.asarray(eo, dtype=int)
            if ei.shape == eo.shape:
                m = (eo == 3) & (ei != 3)
                if m.any():
                    vals, cnt = np.unique(ei[m], return_counts=True)
                    return int(vals[np.argmax(cnt)])
    vals, cnt = np.unique(I, return_counts=True)
    return int(vals[np.argmax(cnt)])


def _deep(I, bgc):
    # cells of the band colour whose whole 8-neighbourhood is band colour too
    # (off-grid counts as open): the "inside" of the empty corridor.
    h, w = I.shape
    P = np.ones((h + 2, w + 2), dtype=bool)
    P[1:-1, 1:-1] = (I == bgc)
    E = np.ones((h, w), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            E &= P[dr:dr + h, dc:dc + w]
    return E


def _runs(rows):
    groups, cur = [], []
    for r in rows:
        if cur and r == cur[-1] + 1:
            cur.append(r)
        else:
            if cur:
                groups.append(cur)
            cur = [r]
    if cur:
        groups.append(cur)
    return groups


def _detect(I, bgc):
    """Return [corridor_cells, arm_cells...] measured from I alone."""
    hi, wi = I.shape
    E = _deep(I, bgc)

    # depth of the open interior measured up from the bottom edge, per column
    dn = np.zeros(wi, dtype=int)
    for c in range(wi):
        r = hi - 1
        while r >= 0 and E[r, c]:
            r -= 1
        dn[c] = hi - 1 - r

    # primary unit: the vertical corridor's interior = largest bottom-anchored rectangle
    best = None
    for a in range(wi):
        m = dn[a]
        for b in range(a, wi):
            m = min(m, dn[b])
            if m == 0:
                break
            area = m * (b - a + 1)
            if best is None or area > best[0]:
                best = (area, a, b, m)
    if best is None:
        return []
    _, ca, cb, m = best
    top = hi - m
    band = [(r, c) for r in range(top, hi) for c in range(ca, cb + 1)]

    # secondary units: arms that leave the corridor sideways and run to the grid edge,
    # each a rectangle from the corridor's side wall out to the border (disjoint from it).
    arms = []
    sides = []
    if ca - 1 >= 0:
        sides.append((0, ca - 1))
    if cb + 1 <= wi - 1:
        sides.append((cb + 1, wi - 1))
    for (c0, c1) in sides:
        rows = [r for r in range(top, hi) if E[r, c0:c1 + 1].all()]
        groups = _runs(rows)
        if groups:
            g = max(groups, key=len)
            arms.append([(r, c) for r in g for c in range(c0, c1 + 1)])
    arms.sort(key=lambda cells: -len(cells))
    return [band] + arms


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = _band_colour(I, examples)
    units = _detect(I, bgc)
    ops, sels = [], []
    if units:
        # primary unit: the vertical corridor's interior
        ops.append(3); sels.append(sel_of(units[0]))
        # then each side arm, from the corridor wall out to the grid edge
        for cells in units[1:]:
            ops.append(3); sels.append(sel_of(cells))
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
