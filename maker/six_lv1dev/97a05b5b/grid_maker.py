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

from maker.sel_helpers import sel_of


# Reading taken from the golf entry: the keys outside the rectangle are only
# references. Crop to the rectangle FIRST. Then match each key's silhouette (its
# rectangle-coloured cells, under any of the 8 rotations/reflections) to a hole
# pattern inside the cropped rectangle, and stamp it there.
# Where the golf entry and the maker agree (no change made): a stamp means "the
# hole is plugged with the rectangle colour, and the rest of the window takes the
# key's colour". Where they differ (followed here): the order is crop first, then
# stamp. Each key's slot is found from I's hole pattern alone, never from O.


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kwargs) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    hlo, wlo = min(15, max_h), min(15, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max(hlo, max_h)))
    w = unifint(diff_lb, diff_ub, (wlo, max(wlo, max_w)))
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


def _components(I, bg):
    hi, wi = I.shape
    seen = np.zeros((hi, wi), dtype=bool)
    out = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bg:
                continue
            seen[r, c] = True
            stack, cells = [(r, c)], []
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < hi and 0 <= ny < wi and not seen[nx, ny] and I[nx, ny] != bg:
                        seen[nx, ny] = True
                        stack.append((nx, ny))
            out.append(cells)
    return out


_DIHEDRAL = [
    lambda A: A, np.fliplr, np.flipud, lambda A: np.rot90(A, 2),
    lambda A: A.T, lambda A: np.rot90(A, 2).T, lambda A: np.rot90(A, 1), lambda A: np.rot90(A, 3),
]


def _read_scene(I, bg):
    """Split I (on background bg) into keys (solid 2-colour rectangles) and the
    bitten rectangle; None if this bg does not fit that reading."""
    keys, rest = [], []
    for cells in _components(I, bg):
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, c0, r1, c1 = min(rs), min(cs), max(rs), max(cs)
        pal = {int(I[p]) for p in cells}
        if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(pal) == 2:
            keys.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
        else:
            rest.extend(cells)
    if not rest:
        return None
    pal = {int(I[p]) for p in rest}
    if len(pal) != 1:
        return None
    sqc = pal.pop()
    R0 = min(p[0] for p in rest); R1 = max(p[0] for p in rest)
    C0 = min(p[1] for p in rest); C1 = max(p[1] for p in rest)
    sub = I[R0:R1 + 1, C0:C1 + 1]
    if not np.all((sub == sqc) | (sub == bg)):
        return None
    return sqc, R0, C0, R1, C1, keys


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    order = [int(v) for v, _ in sorted(zip(*np.unique(I, return_counts=True)), key=lambda t: -t[1])]
    scene, bgc = None, None
    for cand in order:
        scene = _read_scene(I, cand)
        if scene is not None:
            bgc = cand
            break
    ops, sels = [], []
    if scene is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    sqc, R0, C0, R1, C1, keys = scene
    rh, rw = R1 - R0 + 1, C1 - C0 + 1
    holes = (I[R0:R1 + 1, C0:C1 + 1] == bgc)

    # 1. The keys are only references: cut the canvas down to the rectangle first.
    #    (the selection is exactly the full rectangle, so a bbox is right here)
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])

    # 2. Each key's silhouette (its sqc cells, under some rotation/reflection) is one
    #    hole pattern in the rectangle. Key windows never overlap, and together the
    #    silhouettes account for every hole, so solve it as an exact cover (from I only).
    infos, cands = [], []
    for (r0, c0, th, tw) in keys:
        K = I[r0:r0 + th, c0:c0 + tw]
        col = [int(v) for v in np.unique(K) if int(v) != sqc][0]
        mask = (K == sqc)
        lst, seen = [], set()
        for f in _DIHEDRAL:
            M = np.ascontiguousarray(f(mask))
            mh, mw = M.shape
            for rr in range(rh - mh + 1):
                for cc in range(rw - mw + 1):
                    if np.array_equal(holes[rr:rr + mh, cc:cc + mw], M):
                        k = (rr, cc, mh, mw, M.tobytes())
                        if k not in seen:
                            seen.add(k); lst.append((rr, cc, M))
        infos.append(col); cands.append(lst)
    idx = sorted(range(len(keys)), key=lambda i: len(cands[i]))
    used = np.zeros_like(holes)
    choice_ = {}

    def solve(n):
        if n == len(idx):
            return bool(np.array_equal(used & holes, holes))
        i = idx[n]
        for rr, cc, M in cands[i]:
            mh, mw = M.shape
            if np.any(used[rr:rr + mh, cc:cc + mw]):
                continue
            used[rr:rr + mh, cc:cc + mw] = True
            choice_[i] = (rr, cc, M)
            if solve(n + 1):
                return True
            used[rr:rr + mh, cc:cc + mw] = False
        return False

    stamps = []
    if solve(0):
        stamps = [(choice_[i][0], choice_[i][1], choice_[i][2], infos[i]) for i in range(len(keys))]

    # 3. Stamp the keys into the cropped rectangle, one window at a time in raster order.
    #    The hole is plugged with the rectangle colour, and the rest of the window
    #    takes the key's colour.
    for rr, cc, M, col in sorted(stamps, key=lambda s: (s[0], s[1])):
        hole = [(rr + a, cc + b) for a in range(M.shape[0]) for b in range(M.shape[1]) if M[a, b]]
        body = [(rr + a, cc + b) for a in range(M.shape[0]) for b in range(M.shape[1]) if not M[a, b]]
        ops.append(int(sqc)); sels.append(sel_of(hole))
        if body:
            ops.append(int(col)); sels.append(sel_of(body))

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
