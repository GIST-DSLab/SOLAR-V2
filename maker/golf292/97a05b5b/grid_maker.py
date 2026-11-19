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
import numpy as np
import random
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (15, max(15, max_h)))
    w = unifint(diff_lb, diff_ub, (15, max(15, max_w)))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh // 2)
    ow = randint(2, sgw // 2)
    nobjs = unifint(diff_lb, diff_ub, (1, 8))
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
            obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
        obj |= choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
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
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
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
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
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
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {'input': gi, 'output': go}


def _components8(mask):
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            x, y = a + da, b + db
                            if 0 <= x < H and 0 <= y < W and mask[x, y] and not seen[x, y]:
                                seen[x, y] = True
                                stack.append((x, y))
                comps.append(cells)
    return comps


def _syms(t):
    return [t, np.rot90(t, 1), np.rot90(t, 2), np.rot90(t, 3),
            np.fliplr(t), np.flipud(t), t.T, np.rot90(t, 2).T]


def _analyze(I, bgc):
    comps = _components8(I != bgc)
    if not comps:
        return None
    mono, pieces = [], []
    for cells in comps:
        colset = {int(I[r, c]) for r, c in cells}
        (mono if len(colset) == 1 else pieces).append(cells)
    if not mono:
        return None
    # the big rectangle: largest single-colour component (plus same-colour
    # single-colour fragments, in case holes split it)
    big = max(mono, key=len)
    sqc = int(I[big[0]])
    rect_cells = [p for cells in mono if int(I[cells[0]]) == sqc for p in cells]
    rs = [p[0] for p in rect_cells]; cs = [p[1] for p in rect_cells]
    R0, R1, C0, C1 = min(rs), max(rs), min(cs), max(cs)
    S = I[R0:R1 + 1, C0:C1 + 1]
    sh, sw = S.shape
    holes = frozenset(zip(*np.nonzero(S == bgc)))
    cand_lists = []
    for cells in pieces:
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        t = I[min(rs):max(rs) + 1, min(cs):max(cs) + 1]
        cands, seen = [], set()
        for tt in _syms(t):
            th, tw = tt.shape
            pat = np.where(tt == sqc, bgc, sqc)
            for a in range(sh - th + 1):
                for b in range(sw - tw + 1):
                    if np.array_equal(S[a:a + th, b:b + tw], pat):
                        win = frozenset((a + x, b + y) for x in range(th) for y in range(tw))
                        hc = frozenset((a + x, b + y) for x in range(th) for y in range(tw)
                                       if tt[x, y] == sqc)
                        key = (a, b, tt.tobytes(), tt.shape)
                        if key not in seen:
                            seen.add(key)
                            cands.append(((a, b, tt), win, hc))
        if cands:
            cand_lists.append(cands)
    cand_lists.sort(key=len)
    best = [None]

    def dfs(k, used, covered, chosen):
        if k == len(cand_lists):
            if covered == holes:
                best[0] = list(chosen)
                return True
            return False
        for pl, win, hc in cand_lists[k]:
            if win & used:
                continue
            chosen.append(pl)
            if dfs(k + 1, used | win, covered | hc, chosen):
                return True
            chosen.pop()
        return dfs(k + 1, used, covered, chosen)

    dfs(0, frozenset(), frozenset(), [])
    placements = best[0] if best[0] is not None else []
    return R0, C0, R1, C1, sqc, placements


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: the canvas colour the generator paints first; majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    info = _analyze(I, int(bgc))
    ops, sels = [], []
    if info is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    R0, C0, R1, C1, sqc, placements = info
    rh, rw = R1 - R0 + 1, C1 - C0 + 1

    # 1) close every hole of the rectangle in one stroke: all background-coloured
    #    cells inside the rectangle become the rectangle colour.
    holes = [(r, c) for r in range(R0, R1 + 1) for c in range(C0, C1 + 1)
             if I[r, c] == bgc]
    if holes and bgc != sqc:
        ops.append(int(sqc)); sels.append(sel_of(holes))

    # 2) one stroke per key piece: paint its colour on the cells where its
    #    matching-symmetry copy lands around the hole it fits.
    for (a, b, tt) in sorted(placements, key=lambda p: (p[0], p[1])):
        cells = {}
        for x in range(tt.shape[0]):
            for y in range(tt.shape[1]):
                v = int(tt[x, y])
                if v != sqc:
                    cells.setdefault(v, []).append((R0 + a + x, C0 + b + y))
        for v, cl in cells.items():
            ops.append(v); sels.append(sel_of(cl))

    # 3) keep only the rectangle (selection is exactly that full rectangle)
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])
    ops.append(34); sels.append([0, 0, rh - 1, rw - 1])
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
