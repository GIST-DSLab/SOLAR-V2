"""
ARC Task: 97a05b5b (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:                     # pragma: no cover - fallback only
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # bgc / sqc are the two structural roles; the remaining pool feeds the
    # per-piece marker colors (`remcols` in the original generator).
    # 0 is excluded on purpose: ARCLE's object ops (Move / Flip) treat 0 as
    # "nothing there", so a piece containing 0 could not be carried around.
    cols = list(range(1, 10))
    perm = random.sample(cols, len(cols))
    return {"bgc": perm[0], "sqc": perm[1], "objcols": perm[2:]}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sqc=None, objcols=None) -> dict:
    if bgc is None or sqc is None or objcols is None:
        p = random.sample(list(range(1, 10)), 9)
        bgc, sqc, objcols = p[0], p[1], p[2:]

    hhi = max(12, min(30, max_h))
    whi = max(12, min(30, max_w))
    hlo = min(15, hhi)
    wlo = min(15, whi)

    while True:
        h = unifint(diff_lb, diff_ub, (hlo, hhi))
        w = unifint(diff_lb, diff_ub, (wlo, whi))
        sgh = randint(h // 3, h // 3 * 2)
        sgw = randint(w // 3, w // 3 * 2)
        remcols = list(objcols)
        gi = canvas(bgc, (h, w))
        oh = randint(2, max(2, sgh // 2))
        ow = randint(2, max(2, sgw // 2))
        nobjs = unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))

        objs = set()
        cands = asindices(canvas(-1, (oh, ow)))
        forbidden = set()
        tr = 0
        maxtr = 4 * nobjs
        while len(objs) != nobjs and tr < maxtr:
            tr += 1
            obj = {choice(totuple(cands))}
            ncells = randint(1, oh * ow - 1)
            for k in range(ncells - 1):
                opts = (cands - obj) & mapply(neighbors, obj)
                if len(opts) == 0:
                    break
                obj.add(choice(totuple(opts)))
            obj = set(obj) | set(choice((dmirror, cmirror, vmirror, hmirror))(frozenset(obj)))
            obj = frozenset(obj)
            if len(obj) == height(obj) * width(obj):
                continue
            objn = normalize(obj)
            if objn not in forbidden:
                objs.add(objn)
            for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
                for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                    forbidden.add(augmf1(augmf2(objn)))

        tr = 0
        maxtr = 5 * nobjs
        succ = 0
        loci = randint(0, h - sgh)
        locj = randint(0, w - sgw)
        bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
        gi = fill(gi, sqc, bd)
        go = canvas(sqc, (sgh, sgw))
        goinds = asindices(go)
        giinds = asindices(gi) - shift(goinds, (loci, locj))
        giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))

        while succ < nobjs and tr < maxtr and len(objs) > 0 and len(remcols) > 0:
            tr += 1
            obj = choice(totuple(objs))
            col = choice(remcols)
            subgi = fill(canvas(col, shape(obj)), sqc, obj)
            if len(palette(subgi)) == 1:
                continue
            # reflections only -> the piece keeps its footprint, and the
            # re-orientation is expressible as FlipH / FlipV in ARCLE.
            f1 = choice((identity, hmirror, vmirror))
            f2 = choice((identity, hmirror, vmirror))
            f = compose(f1, f2)
            subgo = f(subgi)
            giobj = asobject(subgi)
            goobj = asobject(subgo)
            ohi, owi = shape(giobj)
            oho, owo = shape(goobj)
            gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
            if len(gocands) == 0:
                continue
            goloc = choice(totuple(gocands))
            goplcd = shift(goobj, goloc)
            goplcdi = toindices(goplcd)
            if goplcdi.issubset(goinds):
                gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= w - owi)
                if len(gicands) == 0:
                    continue
                giloc = choice(totuple(gicands))
                giplcd = shift(giobj, giloc)
                giplcdi = toindices(giplcd)
                if giplcdi.issubset(giinds):
                    succ += 1
                    remcols = remove(col, remcols)
                    objs = remove(obj, objs)
                    goinds = goinds - goplcdi
                    giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                    gi = paint(gi, giplcd)
                    gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)),
                                               lambda cij: cij[0] == sqc))
                    go = paint(go, goplcd)

        if succ >= 1:
            return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
def _components(I, bgc):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                comps.append(cells)
    return comps


def _plan(I, O, bgc):
    """Read the rule off I: one big square of `sqc` with `bgc` gaps punched in it,
    plus loose 2-colour rectangular pieces lying around it.  Each piece, once
    reflected, fits exactly into one gap.  Returns the square's origin and, per
    piece, the reflection needed and the gap it belongs to."""
    ho, wo = O.shape
    patches, rest = [], []
    for cells in _components(I, bgc):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cols = {int(I[r, c]) for r, c in cells}
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(cols) == 2:
            patches.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        else:
            rest.extend(cells)
    if not rest or not patches:
        return None
    restcols = {int(I[r, c]) for r, c in rest}
    if len(restcols) != 1:
        return None
    sqc = restcols.pop()
    rs = [r for r, _ in rest]
    cs = [c for _, c in rest]
    R0, C0 = min(rs), min(cs)
    if (max(rs) - R0 + 1, max(cs) - C0 + 1) != (ho, wo):
        return None
    W = np.array(I[R0:R0 + ho, C0:C0 + wo])
    if not set(np.unique(W).tolist()) <= {int(sqc), int(bgc)}:
        return None

    res = W.copy()
    out = []
    for (pr, pc, ph, pw) in patches:
        P = np.array(I[pr:pr + ph, pc:pc + pw])
        pcols = set(np.unique(P).tolist())
        if len(pcols) != 2 or sqc not in pcols:
            return None
        col = [c for c in pcols if c != sqc][0]
        found = None
        # identity first, so a symmetric piece never gets a no-op flip
        for flips, T in ((tuple(), P),
                         ((26,), np.fliplr(P)),
                         ((27,), np.flipud(P)),
                         ((26, 27), np.flipud(np.fliplr(P)))):
            m = (T == sqc)
            gap = np.where(m, bgc, sqc)      # how that gap looks in I
            done = np.where(m, sqc, col)     # how the placed piece looks in O
            th, tw = T.shape
            for dr in range(ho - th + 1):
                for dc in range(wo - tw + 1):
                    if np.array_equal(W[dr:dr + th, dc:dc + tw], gap) and \
                       np.array_equal(O[dr:dr + th, dc:dc + tw], done):
                        found = (flips, dr, dc, th, tw, done)
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            return None
        flips, dr, dc, th, tw, done = found
        res[dr:dr + th, dc:dc + tw] = done
        out.append({'src': (pr, pc, ph, pw), 'flips': flips, 'dr': dr, 'dc': dc})
    if not np.array_equal(res, O):
        return None
    return {'origin': (R0, C0), 'sqc': sqc, 'patches': out}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    plan = None
    for bgc, _ in Counter(I.flatten().tolist()).most_common():
        plan = _plan(I, O, bgc)
        if plan is not None:
            break

    ops, sels = [], []

    if plan is None:
        # safety net (unreachable for grids this generator produces)
        best = None
        for i in range(hi - ho + 1):
            for j in range(wi - wo + 1):
                d = int(np.sum(I[i:i + ho, j:j + wo] != O))
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        ops.append(33); sels.append([i, j, ho - 1, wo - 1])
        cur = I[i:i + ho, j:j + wo]
        for k in range(10):
            cells = [(r, c) for r in range(ho) for c in range(wo)
                     if O[r, c] == k and cur[r, c] != k]
            if cells:
                ops.append(k); sels.append(sel_of(cells))
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    R0, C0 = plan['origin']

    # Handle the pieces in reading order of the gaps they fill.
    for p in sorted(plan['patches'], key=lambda q: (q['dr'], q['dc'])):
        pr, pc, ph, pw = p['src']
        cells = [(r, c) for r in range(pr, pr + ph) for c in range(pc, pc + pw)]

        # (a) reflect the piece where it lies, until it matches its gap
        for fop in p['flips']:
            ops.append(fop); sels.append(sel_of(cells))

        # (b) carry it into the gap, one cell at a time.  The first Move grabs
        #     the piece; every later Move keeps the same grab (empty selection),
        #     so ARCLE restores everything the piece glides over.
        tr, tc = R0 + p['dr'], C0 + p['dc']
        dv, dh = tr - pr, tc - pc
        first = True
        for _ in range(abs(dv)):
            ops.append(21 if dv > 0 else 20)
            sels.append(sel_of(cells) if first else sel_of([]))
            first = False
        for _ in range(abs(dh)):
            ops.append(22 if dh > 0 else 23)
            sels.append(sel_of(cells) if first else sel_of([]))
            first = False

    # (c) keep only the square itself.  Selection is exactly that full
    #     rectangle (background included), so the bbox form is the right one.
    ops.append(33); sels.append([R0, C0, ho - 1, wo - 1])
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 97a05b5b"
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
                                f"for task 97a05b5b"
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
                    f"Failed to build a complete episode for task 97a05b5b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"97a05b5b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
