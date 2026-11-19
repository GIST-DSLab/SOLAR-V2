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
from utils import *
from dsl import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def _background(I):
    # Noise is built so no solid 3x3 block of either colour survives; the only
    # solid 3x3 blocks lie inside the cleared corridors -> their colour is bg.
    h, w = I.shape
    cnt = {}
    for r in range(h - 2):
        for c in range(w - 2):
            blk = I[r:r + 3, c:c + 3]
            v = int(blk[0, 0])
            if (blk == v).all():
                cnt[v] = cnt.get(v, 0) + 1
    if cnt:
        return max(cnt, key=cnt.get)
    vals, n = np.unique(I, return_counts=True)
    return int(vals[np.argmax(n)])


def _runs(flags, min_len):
    out, a = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and a is None:
            a = i
        elif not f and a is not None:
            if i - a >= min_len:
                out.append((a, i - 1))
            a = None
    return out


def _corridors(I):
    """Interior rectangles (r0, c0, r1, c1) of the cleared corridors, from I alone.
    One vertical corridor rises from the bottom edge; one or two horizontal corridors
    leave it toward the left and/or right edge. Interior = corridor minus its 1-cell
    rim against the noise (a side lying on the grid edge has no rim)."""
    h, w = I.shape
    bg = _background(I)
    L = []
    for j in range(w):
        r = h - 1
        while r >= 0 and I[r, j] == bg:
            r -= 1
        L.append(h - 1 - r)
    need = h - h // 2
    runs = _runs([L[j] >= need for j in range(w)], 3)
    best = None
    for (a, b) in runs:
        for x in range(a, b - 1):
            for y in range(x + 2, b + 1):
                key = ((y - x + 1) * min(L[x:y + 1]), y - x)
                if best is None or key > best[0]:
                    best = (key, x, y)
    _, c0, c1 = best
    top = h - min(L[c0:c1 + 1])
    rects = [(top + 1 if top > 0 else 0, c0 + 1, h - 1, c1 - 1)]
    left = [r >= top and bool((I[r, :c0] == bg).all()) for r in range(h)]
    right = [r >= top and bool((I[r, c1 + 1:] == bg).all()) for r in range(h)]
    for (a, b) in _runs(left, 3):
        rects.append((a + 1 if a > 0 else 0, 0, b - 1, c1 - 1))
    for (a, b) in _runs(right, 3):
        rects.append((a + 1 if a > 0 else 0, c0 + 1, b - 1, w - 1))
    return rects


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **_kw) -> dict:
    if bgc is None or noisec is None:
        bgc, noisec = sample([c for c in range(10) if c != 3], 2)
    h = unifint(diff_lb, diff_ub, (18, max(18, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (18, max(18, min(30, max_w))))
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    bgcinds = sample(inds, nbgc)
    gi = fill(gi, bgc, bgcinds)
    sinds = asindices(canvas(-1, (3, 3)))
    bgcf = recolor(bgc, sinds)
    noisecf = recolor(noisec, sinds)
    addn = set()
    addb = set()
    for occ in occurrences(gi, bgcf):
        occi, occj = occ
        addn.add((randint(0, 2) + occi, randint(0, 2) + occj))
    for occ in occurrences(gi, noisecf):
        occi, occj = occ
        addb.add((randint(0, 2) + occi, randint(0, 2) + occj))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)
    go = tuple(e for e in gi)
    dim = randint(randint(3, 8), 8)
    locj = randint(3, w - dim - 4)          # band must fit inside the WIDTH
    spi = choice((0, randint(3, h // 2)))
    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)
    sgns = choice(((-1,), (1,), (-1, 1)))
    startloc = choice((spi, randint(spi + 3, h - 6)))
    hh = randint(3, min(8, h - startloc - 3))
    for sgn in sgns:
        for ii in range(startloc, startloc + hh, 1):
            ln = shoot((ii, locj), (0, sgn))
            gi = fill(gi, bgc, ln)
            go = fill(go, bgc, ln - ofcolor(go, 3))
    for sgn in sgns:
        for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
            ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
            go = fill(go, 3, ln)
    if len(sgns) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
        sgns = (-sgns[0],)
        startloc = choice((spi, randint(spi + 3, h - 6)))
        hh = randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh, 1):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc + 1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                ln = shoot((ii, locj + dim - 2 if sgn == -1 else locj + 1), (0, sgn))
                go = fill(go, 3, ln)
    # reject instances whose corridor layout is ambiguous to read from the input
    # (noise accidentally forming extra corridor-like strips)
    gin = np.array(gi)
    pred = gin.copy()
    for (r0, c0, r1, c1) in _corridors(gin):
        if r0 <= r1 and c0 <= c1:
            pred[r0:r1 + 1, c0:c1 + 1] = 3
    if not (pred == np.array(go)).all():
        raise ValueError("ambiguous corridor layout")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    # The cleared corridor (vertical arm + its horizontal arm(s)) is one connected
    # region measured from I alone; its interior (corridor minus the rim against the
    # noise) is a single connected shape -- the arms overlap inside the vertical
    # band. Paint that whole interior with 3 in one stroke.
    cells = set()
    for (r0, c0, r1, c1) in _corridors(I):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                cells.add((r, c))
    ops.append(3)
    sels.append(sel_of(sorted(cells)))   # corridor interior (connected region)
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
