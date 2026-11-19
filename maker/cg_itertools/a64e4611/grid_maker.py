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
from collections import Counter
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _runs(rows):
    runs, cur = [], []
    for r in rows:
        if cur and r == cur[-1] + 1:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = [r]
    if cur:
        runs.append(cur)
    return runs


def _detect(I, bgc):
    """Find the empty corridors (one vertical shaft from the bottom edge, up to two
    horizontal arms shooting to the side edges) and return each corridor's interior
    rectangle (r0, c0, r1, c1), horizontal arms first, then the shaft."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    B = (I == bgc)
    top = []
    for c in range(w):
        r = h
        while r > 0 and B[r - 1, c]:
            r -= 1
        top.append(r)
    best = None
    for a in range(3, w):
        mx = 0
        for b in range(a, min(w - 4, a + 8)):
            mx = max(mx, top[b])
            if mx >= h:
                break
            if b - a + 1 < 3:
                continue
            key = ((b - a + 1) * (h - mx), h - mx)
            if best is None or key > best[0]:
                best = (key, a, b, mx)
    if best is None:
        return None
    _, a, b, spi = best
    locj, dim = a, b - a + 1

    def longest(rows):
        rs = [r for r in _runs(rows) if len(r) >= 3]
        return max(rs, key=len) if rs else None

    left = longest([r for r in range(spi, h - 3) if B[r, :locj + 1].all()])
    right = longest([r for r in range(spi, h - 3) if B[r, locj:].all()])
    if left and right:
        ls, rs = set(left), set(right)
        ov = ls & rs
        if ov and ls != rs:
            # an overlapping row is more plausibly a real row of the wider arm
            if locj + 1 <= w - locj:
                cand = [r for r in left if r not in ov]
                if len(cand) >= 3 and len(_runs(cand)) == 1:
                    left = cand
            else:
                cand = [r for r in right if r not in ov]
                if len(cand) >= 3 and len(_runs(cand)) == 1:
                    right = cand

    rects = []
    for arm, c0, c1 in ((left, 0, locj + dim - 2), (right, locj + 1, w - 1)):
        if arm:
            s, e = arm[0], arm[-1]
            rects.append((s + 1 if s > 0 else 0, c0, e - 1, c1))
    rects.append((spi + 1 if spi > 0 else 0, locj + 1, h - 1, locj + dim - 2))
    return rects


def _render(I, rects):
    G = np.array(I, dtype=int)
    for (r0, c0, r1, c1) in rects:
        G[r0:r1 + 1, c0:c1 + 1] = 3
    return G


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def _gen_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec):
    h = _unifint(diff_lb, diff_ub, (18, max(18, max_h)))
    w = _unifint(diff_lb, diff_ub, (18, max(18, max_w)))
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = _unifint(diff_lb, diff_ub, (lb, ub))
    gi = np.full((h, w), noisec, dtype=int)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for (i, j) in random.sample(inds, nbgc):
        gi[i, j] = bgc
    addn, addb = set(), set()
    for i in range(h - 2):
        for j in range(w - 2):
            win = gi[i:i + 3, j:j + 3]
            if (win == bgc).all():
                addn.add((i + random.randint(0, 2), j + random.randint(0, 2)))
            elif (win == noisec).all():
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
    go[(spi + 1 if spi > 0 else spi):, locj + 1:locj + dim - 1] = 3

    def arms(sgns, startloc, hh):
        for sgn in sgns:
            for ii in range(startloc, startloc + hh):
                cs = range(0, locj + 1) if sgn == -1 else range(locj, w)
                for j in cs:
                    gi[ii, j] = bgc
                    if go[ii, j] != 3:
                        go[ii, j] = bgc
        for sgn in sgns:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1):
                if sgn == -1:
                    go[ii, 0:locj + dim - 1] = 3
                else:
                    go[ii, locj + 1:] = 3

    sgns = random.choice(((-1,), (1,), (-1, 1)))
    startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
    hh = random.randint(3, min(8, h - startloc - 3))
    arms(sgns, startloc, hh)
    if len(sgns) == 1 and _unifint(diff_lb, diff_ub, (0, 1)) == 1:
        sgns = (-sgns[0],)
        startloc = random.choice((spi, random.randint(spi + 3, h - 6)))
        hh = random.randint(3, min(8, h - startloc - 3))
        arms(sgns, startloc, hh)
    return gi, go


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, noisec=None, **kwargs) -> dict:
    if bgc is None or noisec is None:
        c = sample_colors()
        bgc = c["bgc"] if bgc is None else bgc
        noisec = c["noisec"] if noisec is None else noisec
    last = None
    for _ in range(100):
        gi, go = _gen_once(diff_lb, diff_ub, max_h, max_w, bgc, noisec)
        last = (gi, go)
        rects = _detect(gi, bgc)
        # keep only instances whose corridors are unambiguous from the input alone
        if rects is not None and np.array_equal(_render(gi, rects), go):
            break
    gi, go = last
    return {"input": tuple(tuple(int(v) for v in r) for r in gi),
            "output": tuple(tuple(int(v) for v in r) for r in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the colour the corridors are cut from; the demonstrations show it
    # (whatever an example turns into 3 was background there).
    bgc = None
    for ex in (examples or []):
        EI = np.asarray(ex[0], dtype=int)
        EO = np.asarray(ex[1], dtype=int)
        if EI.shape == EO.shape:
            m = (EO == 3) & (EI != 3)
            if m.any():
                bgc = Counter(EI[m].tolist()).most_common(1)[0][0]
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # corridors measured from I alone: horizontal arms first, then the vertical shaft
    rects = _detect(I, bgc) or []

    G = I.copy()
    ops, sels = [], []

    def paint(cells):
        cells = [(r, c) for (r, c) in cells]
        if cells and any(G[r, c] != 3 for r, c in cells):
            ops.append(3)
            sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = 3

    # stage 1: each corridor filled with 3 while keeping a one-cell background
    # margin on every side (the eroded core of the empty band)
    for (r0, c0, r1, c1) in rects:
        core = [(r, c) for r in range(max(r0, 1), min(r1, hi - 2) + 1)
                for c in range(max(c0, 1), min(c1, wi - 2) + 1)]
        paint(core)
    # stage 2: where a corridor runs into the grid border, the fill carries on
    # through the margin out to the edge
    for (r0, c0, r1, c1) in rects:
        edge = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if r in (0, hi - 1) or c in (0, wi - 1)]
        paint(edge)

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
