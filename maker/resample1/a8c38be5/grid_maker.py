"""
ARC Task: a8c38be5 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None or sqc is None:
        bgc, sqc = sample(cols, 2)
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    goh = unifint(diff_lb, diff_ub, (9, max(9, min(20, max_h - 4))))
    gow = unifint(diff_lb, diff_ub, (9, max(9, min(20, max_w - 4))))
    h = unifint(diff_lb, diff_ub, (goh + 4, max(goh + 4, max_h)))
    w = unifint(diff_lb, diff_ub, (gow + 4, max(gow + 4, max_w)))
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh - 7)
    loci2 = randint(loci1 + 4, goh - 3)
    locj1 = randint(2, gow - 7)
    locj2 = randint(locj1 + 4, gow - 3)
    fs = hfrontier((loci1, 0)) | hfrontier((loci2, 0)) | vfrontier((0, locj1)) | vfrontier((0, locj2))
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    hdelt = h - goh
    hdelt1 = randint(1, hdelt - 3)
    hdelt2 = randint(1, hdelt - hdelt1 - 2)
    hdelt3 = randint(1, hdelt - hdelt1 - hdelt2 - 1)
    hdelt4 = hdelt - hdelt1 - hdelt2 - hdelt3
    wdelt = w - gow
    wdelt1 = randint(1, wdelt - 3)
    wdelt2 = randint(1, wdelt - wdelt1 - 2)
    wdelt3 = randint(1, wdelt - wdelt1 - wdelt2 - 1)
    wdelt4 = wdelt - wdelt1 - wdelt2 - wdelt3
    gi = gi[:loci2] + repeat(repeat(bgc, gow), hdelt2) + gi[loci2:]
    gi = gi[:loci1 + 1] + repeat(repeat(bgc, gow), hdelt3) + gi[loci1 + 1:]
    gi = repeat(repeat(bgc, gow), hdelt1) + gi + repeat(repeat(bgc, gow), hdelt4)
    gi = dmirror(gi)
    gi = gi[:locj2] + repeat(repeat(bgc, h), wdelt2) + gi[locj2:]
    gi = gi[:locj1 + 1] + repeat(repeat(bgc, h), wdelt3) + gi[locj1 + 1:]
    gi = repeat(repeat(bgc, h), wdelt1) + gi + repeat(repeat(bgc, h), wdelt4)
    gi = dmirror(gi)
    nswitcheroos = unifint(diff_lb, diff_ub, (0, 10))
    if choice((True, False)):
        gi = gi[loci1 + hdelt1 + 1:] + gi[:loci1 + hdelt1 + 1]
    if choice((True, False)):
        gi = dmirror(gi)
        gi = gi[locj1 + wdelt1 + 1:] + gi[:locj1 + wdelt1 + 1]
        gi = dmirror(gi)
    for k in range(nswitcheroos):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, T, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        mpr = dict()
        for obj in objs:
            shp = shape(obj)
            mpr.setdefault(shp, []).append(obj)
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = choice(ress)
        a, b = sample(abc, 2)
        ap = shift(normalize(a), ulcorner(b))
        bp = shift(normalize(b), ulcorner(a))
        gi = paint(gi, ap | bp)
    nshifts = unifint(diff_lb, diff_ub, (0, 30))
    for k in range(nshifts):
        o = asobject(gi)
        tmpc = canvas(bgc, (h + 12, w + 12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, F, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        objs = sfilter(objs, compose(flip, rbind(bordering, gi)))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        direc1 = (randint(-1, 1), randint(-1, 1))
        direc2 = position({(h // 2, w // 2)}, {center(obj)})
        direc = choice((direc1, direc2))
        gi = fill(gi, bgc, obj)
        gi = paint(gi, shift(obj, direc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    # keep only instances whose 8 frame segments survive intact and assemble to go
    if _assemble(np.array(gi), bgc, sqc) != np.array(go).tolist():
        raise ValueError("pieces collided; retry")
    return {"input": gi, "output": go}


def _components(I, colors_excl):
    hi, wi = I.shape
    seen = np.zeros_like(I, dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or int(I[r, c]) in colors_excl:
                continue
            col = int(I[r, c])
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                y, x = q.popleft(); cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and int(I[ny, nx]) == col:
                        seen[ny, nx] = True; q.append((ny, nx))
            comps.append((col, cells))
    return comps


def _classify(cells):
    s = set(cells)
    r0 = min(r for r, _ in s); r1 = max(r for r, _ in s)
    c0 = min(c for _, c in s); c1 = max(c for _, c in s)
    top = {(r0, c) for c in range(c0, c1 + 1)}
    bot = {(r1, c) for c in range(c0, c1 + 1)}
    lef = {(r, c0) for r in range(r0, r1 + 1)}
    rig = {(r, c1) for r in range(r0, r1 + 1)}
    if s == top | lef: return "TL"
    if s == top | rig: return "TR"
    if s == bot | lef: return "BL"
    if s == bot | rig: return "BR"
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    def notch(full, other_side):
        rest = s - full
        return full <= s and len(rest) == 1 and rest <= other_side and not (rest & corners)
    if r1 - r0 == 1 and notch(top, bot): return "T"
    if r1 - r0 == 1 and notch(bot, top): return "B"
    if c1 - c0 == 1 and notch(lef, rig): return "L"
    if c1 - c0 == 1 and notch(rig, lef): return "R"
    return None


def _segments(I, bgc, sqc):
    segs = {}
    for col, cells in _components(I, {bgc, sqc}):
        k = _classify(cells)
        if k is not None and k not in segs:
            r0 = min(r for r, _ in cells); c0 = min(c for _, c in cells)
            hh = max(r for r, _ in cells) - r0 + 1; ww = max(c for _, c in cells) - c0 + 1
            segs[k] = dict(col=col, cells=cells, r=r0, c=c0, h=hh, w=ww)
    return segs


def _layout(segs):
    S = segs
    ho = S["TL"]["h"] + S["BL"]["h"] + S["L"]["h"] + 2
    wo = S["TL"]["w"] + S["TR"]["w"] + S["T"]["w"] + 2
    dest = {
        "TL": (0, 0),
        "T": (0, S["TL"]["w"] + 1),
        "TR": (0, wo - S["TR"]["w"]),
        "R": (S["TR"]["h"] + 1, wo - S["R"]["w"]),
        "BR": (ho - S["BR"]["h"], wo - S["BR"]["w"]),
        "B": (ho - S["B"]["h"], S["BL"]["w"] + 1),
        "BL": (ho - S["BL"]["h"], 0),
        "L": (S["TL"]["h"] + 1, 0),
    }
    return ho, wo, dest


ORDER = ["TL", "T", "TR", "R", "BR", "B", "BL", "L"]


def _assemble(I, bgc, sqc):
    try:
        segs = _segments(I, bgc, sqc)
        ho, wo, dest = _layout(segs)
    except KeyError:
        return None
    G = np.full((ho, wo), sqc, dtype=int)
    for k in ORDER:
        s = segs[k]; dr, dc = dest[k]
        for r, c in s["cells"]:
            rr, cc = r - s["r"] + dr, c - s["c"] + dc
            if not (0 <= rr < ho and 0 <= cc < wo):
                return None
            G[rr, cc] = s["col"]
    return G.tolist()


def derive_operations(I, O, examples=None):
    from collections import Counter
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cI = Counter(I.flatten().tolist())
    bgc = sqc = None
    if examples:
        exs = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in examples]
        out_cols = set()
        out_cnt = Counter()
        for _, eo in exs:
            out_cols |= set(eo.flatten().tolist())
            out_cnt.update(eo.flatten().tolist())
        # background: present in every input, never part of any assembled output
        cand = [c for c in cI if c not in out_cols and all((ei == c).any() for ei, _ in exs)]
        if cand:
            bgc = max(cand, key=lambda c: cI[c])
        # fill colour: dominates the assembled outputs
        sq = [c for c, _ in out_cnt.most_common() if c != bgc and c in cI]
        if sq:
            sqc = sq[0]
    if bgc is None:
        bgc = cI.most_common(1)[0][0]
    if sqc is None or sqc == bgc:
        cc = Counter(cI); del cc[bgc]
        sqc = cc.most_common(1)[0][0]
    segs = _segments(I, bgc, sqc)
    ho, wo, dest = _layout(segs)

    ops, sels = [], []
    # 1. new canvas of the frame's size; lay the fill colour as base over the whole canvas
    ops.append(33); sels.append([0, 0, ho - 1, wo - 1])  # full rectangle canvas resize
    G = np.zeros((ho, wo), dtype=int)
    sub = I[:min(ho, hi), :min(wo, wi)]
    G[:sub.shape[0], :sub.shape[1]] = sub
    if (G != sqc).any():
        ops.append(sqc); sels.append([0, 0, ho - 1, wo - 1])  # whole canvas = fill colour
        G[:, :] = sqc
    # 2. bring each frame segment (its full bbox rectangle, segment + fill) to its slot
    for k in ORDER:
        s = segs[k]; dr, dc = dest[k]
        box = I[s["r"]:s["r"] + s["h"], s["c"]:s["c"] + s["w"]]
        tgt = G[dr:dr + s["h"], dc:dc + s["w"]]
        nz = box != 0
        if (tgt[nz] != box[nz]).any():
            ops.append(28); sels.append([s["r"], s["c"], s["h"] - 1, s["w"] - 1])  # segment bbox rect
            ops.append(30); sels.append([dr, dc, 0, 0])
            tgt[nz] = box[nz]
        if s["col"] == 0:
            # colour 0 is not carried by Copy/Paste: paint the segment's cells explicitly
            cells = [(r - s["r"] + dr, c - s["c"] + dc) for r, c in s["cells"]]
            if any(G[r, c] != 0 for r, c in cells):
                ops.append(0); sels.append(sel_of(cells))
                for r, c in cells:
                    G[r, c] = 0
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
                        f"num_examples+1 ({num_examples + 1}) for task a8c38be5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a8c38be5"
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
                                f"for task a8c38be5"
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
                    f"Failed to build a complete episode for task a8c38be5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a8c38be5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
