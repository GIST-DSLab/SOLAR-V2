"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    """Episode-level colours: the 'erased block' colour (fixc) and the ordered
    palette the periodic pattern draws its colours from.  Also pre-plans the
    discrete structural variants (straight stripes vs diagonal stripes, and
    whether the whole thing is rotated 90 degrees)."""
    cols = list(range(10))
    fixc = random.choice(cols)
    rest = [c for c in cols if c != fixc]
    random.shuffle(rest)

    VARIANTS = [
        {"straight": True,  "rotated": False},
        {"straight": True,  "rotated": True},
        {"straight": False, "rotated": False},
        {"straight": False, "rotated": True},
    ]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"fixc": fixc, "ccols_pool": rest, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             fixc=None, ccols_pool=None, straight=None, rotated=None) -> dict:
    if fixc is None:
        fixc = choice(interval(0, 10, 1))
    if ccols_pool is None:
        ccols_pool = [c for c in interval(0, 10, 1) if c != fixc]
        shuffle(ccols_pool)
        ccols_pool = list(ccols_pool)
    if straight is None:
        straight = choice((True, False))
    if rotated is None:
        rotated = choice((True, False))

    hi_h = max(7, max_h)
    hi_w = max(7, max_w)
    h = unifint(diff_lb, diff_ub, (7, hi_h))
    w = unifint(diff_lb, diff_ub, (7, hi_w))
    p = unifint(diff_lb, diff_ub, (2, max(2, min(9, min(h // 3, w // 3)))))
    p = min(p, len(ccols_pool))

    ccols = list(ccols_pool[:p])
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {e for e in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if straight:
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)

    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if len(occurrences(tmpg, baseobj)) > 1 and \
           len([r for r in tmpg if fixc not in r]) > 0 and \
           len([r for r in dmirror(tmpg) if fixc not in r]) > 0:
            gi = tmpg
            succ += 1
        tr += 1

    if rotated:
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """The grid is one periodic pattern with rectangular blocks erased in a
    single 'noise' colour.  Measure the two periods from the rows/columns that
    survived intact, take one undamaged period-tile with CopyI, and Paste it
    over each hole at the lattice offsets that cover it.  Cells of the period
    whose colour is 0 are not carried by Paste -> put them back with Color0."""
    import numpy as np
    from collections import Counter, deque
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(r), int(c)) for r, c in cells]}

    A = np.asarray(I, dtype=int)
    h, w = A.shape
    SUB = [0, 0, h - 1, w - 1]          # bbox == whole grid, exactly the region meant
    ops, sels = [], []

    maxpv = max(1, min(9, h // 3))      # generator: period <= min(h,w)//3 and <= 9
    maxph = max(1, min(9, w // 3))

    def analyse(k):
        """Assume k is the erase colour; recover (pv, ph, tile) or None."""
        crows = [r for r in range(h) if not (A[r] == k).any()]
        ccls = [c for c in range(w) if not (A[:, c] == k).any()]
        if not crows or not ccls:
            return None
        ph = None
        for d in range(1, maxph + 1):
            if all(A[r, c] == A[r, c + d] for r in crows for c in range(w - d)):
                ph = d
                break
        if ph is None:
            return None
        pv = None
        for d in range(1, maxpv + 1):
            if all(A[r, c] == A[r + d, c] for c in ccls for r in range(h - d)):
                pv = d
                break
        if pv is None:
            return None
        T = [[-1] * ph for _ in range(pv)]
        for r in range(h):
            for c in range(w):
                v = int(A[r, c])
                if v == k:
                    continue
                a, b = r % pv, c % ph
                if T[a][b] < 0:
                    T[a][b] = v
                elif T[a][b] != v:
                    return None
        if any(T[a][b] < 0 for a in range(pv) for b in range(ph)):
            return None
        return pv, ph, T

    found = None
    for k, _cnt in sorted(Counter(A.flatten().tolist()).items(), key=lambda kv: kv[1]):
        res = analyse(k)
        if res is not None:
            found = (k, res[0], res[1], res[2])
            break

    if found is None:                    # nothing was erased: pattern already whole
        ops.append(34); sels.append(SUB)
        return ops, sels

    k, pv, ph, T = found
    tgt = [[T[r % pv][c % ph] for c in range(w)] for r in range(h)]
    bad = (A == k)

    # ---- pick an undamaged period-tile, preferring one aligned to the lattice ----
    src = None
    for r0 in range(0, h - pv + 1):
        for c0 in range(0, w - ph + 1):
            if r0 % pv or c0 % ph:
                continue
            if not bad[r0:r0 + pv, c0:c0 + ph].any():
                src = (r0, c0)
                break
        if src is not None:
            break
    if src is None:
        for r0 in range(0, h - pv + 1):
            for c0 in range(0, w - ph + 1):
                if not bad[r0:r0 + pv, c0:c0 + ph].any():
                    src = (r0, c0)
                    break
            if src is not None:
                break

    # ---- holes as connected regions ----
    seen = np.zeros((h, w), dtype=bool)
    holes = []
    for r in range(h):
        for c in range(w):
            if bad[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and bad[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                holes.append(cells)
    holes.sort(key=lambda cs: (min(y for y, _ in cs), min(x for _, x in cs)))

    sim = A.copy()
    body_ops, body_sels = [], []
    used_paste = False

    for cells in holes:
        r1 = min(y for y, _ in cells); r2 = max(y for y, _ in cells)
        c1 = min(x for _, x in cells); c2 = max(x for _, x in cells)

        if src is not None:
            ra, ca = src[0] % pv, src[1] % ph
            rows = [R for R in range(ra, h, pv) if R <= r2 and R + pv - 1 >= r1]
            cls_ = [C for C in range(ca, w, ph) if C <= c2 and C + ph - 1 >= c1]
            for R in rows:
                for C in cls_:
                    changed = False
                    for i in range(pv):
                        for j in range(ph):
                            y, x = R + i, C + j
                            if y >= h or x >= w:
                                continue
                            v = T[(R + i) % pv][(C + j) % ph]
                            if v == 0:          # Paste is transparent on 0
                                continue
                            if sim[y, x] != v:
                                sim[y, x] = v
                                changed = True
                    if changed:
                        body_ops.append(30)
                        body_sels.append([R, C, 0, 0])   # paste origin only
                        used_paste = True

        # whatever the paste could not carry (colour-0 cells of the period, and
        # any cell outside the lattice reach) is painted back per colour
        left = {}
        for (y, x) in cells:
            v = tgt[y][x]
            if sim[y, x] != v:
                left.setdefault(v, []).append((y, x))
        for v in sorted(left):
            pts = left[v]
            body_ops.append(int(v))
            body_sels.append(sel_of(pts))
            for (y, x) in pts:
                sim[y, x] = v

    if used_paste:
        # full rectangle: exactly one period of the intact pattern
        ops.append(28); sels.append([src[0], src[1], pv - 1, ph - 1])
    ops.extend(body_ops); sels.extend(body_sels)
    ops.append(34); sels.append(SUB)
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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
