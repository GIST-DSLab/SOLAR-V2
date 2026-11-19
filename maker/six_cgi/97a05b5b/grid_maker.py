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
from collections import Counter
import numpy as np
from maker.sel_helpers import sel_of
from utils import *
from dsl import *


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (min(15, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(15, max_w), max_w))
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
    return {"input": gi, "output": go}


def _components(I, bgc):
    hi, wi = I.shape
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] != bgc:
                        seen[x, y] = True
                        stack.append((x, y))
            comps.append(cells)
    return comps


_DIHEDRAL = [
    lambda g: g, lambda g: np.rot90(g, 1), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 3),
    lambda g: np.fliplr(g), lambda g: np.flipud(g), lambda g: g.T, lambda g: np.rot90(g, 2).T,
]


def _analyze(I, bgc):
    comps = _components(I, bgc)
    keys, rest = [], []
    for cells in comps:
        rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        ncol = len({int(I[p]) for p in cells})
        if len(cells) == area and ncol == 2:
            keys.append((r0, c0, I[r0:r1 + 1, c0:c1 + 1].copy()))
        else:
            rest.extend(cells)
    if not rest:
        return None
    sq_colors = {int(I[p]) for p in rest}
    if len(sq_colors) != 1:
        return None
    sqc = sq_colors.pop()
    R0 = min(p[0] for p in rest); R1 = max(p[0] for p in rest)
    C0 = min(p[1] for p in rest); C1 = max(p[1] for p in rest)
    rect = I[R0:R1 + 1, C0:C1 + 1]
    holes = (rect == bgc)
    rh, rw = rect.shape
    # candidate placements per key: every orientation whose sqc-pattern fits a hole exactly
    cand = []
    for ki, (kr, kc, K) in enumerate(keys):
        kcols = set(np.unique(K).tolist())
        if sqc not in kcols:
            return None
        col = (kcols - {sqc}).pop()
        opts = []
        seen = set()
        for f in _DIHEDRAL:
            T = f(K)
            sig = (T.shape, T.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            M = (T == sqc)
            th, tw = M.shape
            for r in range(rh - th + 1):
                for c in range(rw - tw + 1):
                    if np.array_equal(holes[r:r + th, c:c + tw], M):
                        opts.append((r, c, M))
        if not opts:
            return None
        cand.append((ki, col, opts))
    order = sorted(range(len(cand)), key=lambda i: len(cand[i][2]))
    used = np.zeros((rh, rw), dtype=bool)
    chosen = {}

    # windows are disjoint; together their sqc-patterns must cover every hole exactly
    def bt2(k):
        if k == len(order):
            cov = np.zeros((rh, rw), dtype=bool)
            for i, (r, c, M) in chosen.items():
                cov[r:r + M.shape[0], c:c + M.shape[1]] |= M
            return bool(np.array_equal(cov, holes))
        i = order[k]
        for (r, c, M) in cand[i][2]:
            th, tw = M.shape
            if used[r:r + th, c:c + tw].any():
                continue
            used[r:r + th, c:c + tw] = True
            chosen[i] = (r, c, M)
            if bt2(k + 1):
                return True
            used[r:r + th, c:c + tw] = False
            del chosen[i]
        return False

    if not bt2(0):
        return None
    placements = []
    for i in range(len(cand)):
        ki, col, _ = cand[i]
        r, c, M = chosen[i]
        placements.append((keys[ki][0], keys[ki][1], r, c, M, col))
    return R0, C0, R1, C1, sqc, placements


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    info = None
    for cand_bg, _ in Counter(I.flatten().tolist()).most_common():
        info = _analyze(I, int(cand_bg))
        if info is not None:
            break
    ops, sels = [], []
    if info is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    R0, C0, R1, C1, sqc, placements = info
    rh, rw = R1 - R0 + 1, C1 - C0 + 1
    # 1. Keep only the big rectangle (selection is exactly that full rectangle).
    ops.append(33); sels.append([R0, C0, rh - 1, rw - 1])
    # 2. Take the keys one at a time (in the order they sit in the input) and stamp each
    #    onto the hole its pattern fits: the hole cells (still showing background in the
    #    cropped rectangle) become the rectangle colour, the rest of that key-sized
    #    window takes the key's own colour.
    for (kr, kc, r, c, M, col) in sorted(placements, key=lambda p: (p[0], p[1])):
        th, tw = M.shape
        hole = [(r + a, c + b) for a in range(th) for b in range(tw) if M[a, b]]
        body = [(r + a, c + b) for a in range(th) for b in range(tw) if not M[a, b]]
        if hole:
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
