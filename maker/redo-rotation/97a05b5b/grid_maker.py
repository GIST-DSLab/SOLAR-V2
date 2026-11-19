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


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    """bgc = canvas background, sqc = colour of the big rectangle,
    objcols = pool the little 2-colour tiles draw their colour from.
    Colour 0 is excluded on purpose: ARCLE treats 0 as 'nothing' for
    Move/Copy, and the trajectory physically moves the tiles."""
    cols = [c for c in range(1, 10)]
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    rest = [c for c in cols if c not in (bgc, sqc)]
    random.shuffle(rest)
    return {"bgc": bgc, "sqc": sqc, "objcols": tuple(rest)}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, objcols=None,
             **kwargs) -> dict:
    allcols = [c for c in range(1, 10)]
    if bgc is None:
        bgc = choice(allcols)
    if sqc is None:
        sqc = choice([c for c in allcols if c != bgc])
    if objcols is None:
        objcols = tuple(c for c in allcols if c not in (bgc, sqc))

    hlo = min(15, max_h)
    wlo = min(15, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max(hlo, max_h)))
    w = unifint(diff_lb, diff_ub, (wlo, max(wlo, max_w)))
    sgh = randint(h // 3, h // 3 * 2)
    sgw = randint(w // 3, w // 3 * 2)
    remcols = [c for c in objcols]
    gi = canvas(bgc, (h, w))
    oh = randint(2, max(2, sgh // 2))
    ow = randint(2, max(2, sgw // 2))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, min(8, len(remcols)))))

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
    while succ < nobjs and tr < maxtr and len(objs) > 0 and len(remcols) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        # only the mirror group that ARCLE can perform on any rectangle
        # (FlipH / FlipV / FlipH+FlipV == rot180)
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
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)),
                                           lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)

    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. trajectory
# ----------------------------------------------------------------------------
def derive_operations(I, O):
    """Rule: the big rectangle carries bgc 'bite marks'.  Every bite mark is the
    silhouette of one of the little 2-colour tiles lying outside, mirrored.
    So: mirror each tile (FlipH / FlipV / both) and slide it into its slot,
    then crop to the rectangle."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # ---- connected components of non-background cells (4-connectivity) ----
    def components(bg):
        seen = np.zeros((hi, wi), dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if seen[r, c] or int(I[r, c]) == bg:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < hi and 0 <= ny < wi and not seen[nx, ny] \
                                and int(I[nx, ny]) != bg:
                            seen[nx, ny] = True
                            stack.append((nx, ny))
                out.append(cells)
        return out

    # ---- split into (solid 2-colour tiles) vs (the bitten rectangle) ----
    def analyze(bg):
        tiles, others = [], []
        for cells in components(bg):
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            pal = set(int(I[p[0], p[1]]) for p in cells)
            if len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1) and len(pal) == 2:
                tiles.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
            else:
                others.append((r0, c0, r1, c1))
        if not others:
            return None
        r0 = min(o[0] for o in others)
        c0 = min(o[1] for o in others)
        r1 = max(o[2] for o in others)
        c1 = max(o[3] for o in others)
        if (r1 - r0 + 1, c1 - c0 + 1) != (ho, wo):
            return None
        sub = I[r0:r1 + 1, c0:c1 + 1]
        pal = set(int(v) for v in sub.flatten().tolist()) - {bg}
        if len(pal) != 1:
            return None
        sq = pal.pop()
        if bool(np.any((sub == bg) & (O != sq))):
            return None
        if bool(np.any((O != sq) & (sub != sq))):
            return None
        return bg, sq, r0, c0, tiles

    vals, cnts = np.unique(I, return_counts=True)
    cand_bg = [int(v) for _, v in sorted(zip(cnts.tolist(), vals.tolist()),
                                         key=lambda t: -t[0])]
    info = None
    for bg in cand_bg:
        info = analyze(bg)
        if info is not None:
            break

    ops, sels = [], []

    if info is None:  # emergency path, never reached for this generator
        if (hi, wi) != (ho, wo):
            ops.append(33)
            sels.append([0, 0, ho - 1, wo - 1])
        groups = {}
        for r in range(ho):
            for c in range(wo):
                groups.setdefault(int(O[r, c]), []).append((r, c))
        for colr in sorted(groups):
            ops.append(colr)
            sels.append(sel_of(groups[colr]))
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    bgc, sqc, sr, sc, tiles = info
    sub = I[sr:sr + ho, sc:sc + wo]

    # mirror group; the last field is the ARCLE op chain that performs it
    TRANSFORMS = [
        (lambda A: np.array(A), []),                 # identity
        (lambda A: np.fliplr(A), [26]),              # vmirror  (FlipH)
        (lambda A: np.flipud(A), [27]),              # hmirror  (FlipV)
        (lambda A: np.rot90(A, 2), [26, 27]),        # rot180
        (lambda A: np.rot90(A, 3), None),            # rot90 cw   (dims swap)
        (lambda A: np.rot90(A, 1), None),            # rot90 ccw  (dims swap)
        (lambda A: np.array(A).T, None),             # dmirror
        (lambda A: np.rot90(A, 2).T, None),          # cmirror
    ]

    # ---- match every tile against the bite marks in the rectangle ----
    plans = []
    for (r0, c0, th, tw) in tiles:
        G = I[r0:r0 + th, c0:c0 + tw]
        found = None
        for fn, fops in TRANSFORMS:
            T = np.array(fn(G))
            ph, pw = T.shape
            if ph > ho or pw > wo:
                continue
            bitten = np.where(T == sqc, bgc, sqc)   # how that tile carved the rect
            for rr in range(ho - ph + 1):
                for cc in range(wo - pw + 1):
                    if np.array_equal(sub[rr:rr + ph, cc:cc + pw], bitten) and \
                            np.array_equal(O[rr:rr + ph, cc:cc + pw], T):
                        found = (rr, cc, T, fops)
                        break
                if found is not None:
                    break
            if found is not None:
                break
        if found is not None:
            plans.append((r0, c0, th, tw, found))
    plans.sort(key=lambda p: (p[4][0], p[4][1]))

    W = I.copy()  # working grid, kept in sync with the emitted ops

    for (r0, c0, th, tw, (rr, cc, T, fops)) in plans:
        cells = [(r, c) for r in range(r0, r0 + th) for c in range(c0, c0 + tw)]
        dest_r, dest_c = sr + rr, sc + cc
        ph, pw = T.shape
        movable = fops is not None and 0 not in set(int(v) for v in T.flatten().tolist())

        if movable:
            # (a) mirror the tile where it lies (selection = the tile's own cells)
            for op in fops:
                ops.append(op)
                sels.append(sel_of(cells))
            W[r0:r0 + th, c0:c0 + tw] = T

            # (b) slide it into its slot: first Move grabs, the rest continue it
            dr = dest_r - r0
            dc = dest_c - c0
            first = True
            for _ in range(abs(dr)):
                ops.append(21 if dr > 0 else 20)
                sels.append(sel_of(cells) if first else sel_of([]))
                first = False
            for _ in range(abs(dc)):
                ops.append(22 if dc > 0 else 23)
                sels.append(sel_of(cells) if first else sel_of([]))
                first = False
            if not first:
                W[r0:r0 + th, c0:c0 + tw] = 0      # ARCLE leaves the footprint at 0
                W[dest_r:dest_r + ph, dest_c:dest_c + pw] = T
            # no footprint repair: those cells lie outside the final crop
        else:
            # transposing mirror (or a 0-coloured tile): paint the slot instead
            groups = {}
            for i in range(ph):
                for j in range(pw):
                    tgt = int(T[i, j])
                    ar, ac = dest_r + i, dest_c + j
                    if int(W[ar, ac]) != tgt:
                        groups.setdefault(tgt, []).append((ar, ac))
            for colr in sorted(groups):
                ops.append(colr)
                sels.append(sel_of(groups[colr]))
                for (ar, ac) in groups[colr]:
                    W[ar, ac] = colr

    # ---- safety net (only fires if some tile found no slot) ----
    cur = W[sr:sr + ho, sc:sc + wo]
    if bool(np.any(cur != O)):
        groups = {}
        for r in range(ho):
            for c in range(wo):
                if int(cur[r, c]) != int(O[r, c]):
                    groups.setdefault(int(O[r, c]), []).append((sr + r, sc + c))
        for colr in sorted(groups):
            ops.append(colr)
            sels.append(sel_of(groups[colr]))
            for (ar, ac) in groups[colr]:
                W[ar, ac] = colr

    # ---- keep only the rectangle (bbox == exactly the region meant) ----
    if (sr, sc) != (0, 0) or (ho, wo) != (hi, wi):
        ops.append(33)
        sels.append([sr, sc, ho - 1, wo - 1])

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
