"""
ARC Task: 8a004b2b (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # every colour role the generator samples: background, frame corners,
    # the two marker pixels, and the body colour of the key picture
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, cornc=4, ac1=1, ac2=2, objc=3) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, h // 5))
    ow = unifint(diff_lb, diff_ub, (2, w // 5))
    bounds = asindices(canvas(-1, (oh, ow)))
    gi = canvas(bgc, (h, w))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fp1 = choice(totuple(obj))
    fp2 = choice(remove(fp1, totuple(obj)))
    remobj = obj - {fp1, fp2}
    obj = recolor(objc, remobj) | {(ac1, fp1), (ac2, fp2)}
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    hscf = unifint(diff_lb, diff_ub, (1, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (1, maxwscf))
    loci = randint(0, 2)
    locj = randint(0, 2)
    oplcd = shift(obj, (loci, locj))
    gi = paint(gi, oplcd)
    inh = hscf * oh
    inw = wscf * ow
    sqh = unifint(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = unifint(diff_lb, diff_ub, (inw + 2, w))
    sqloci = randint(loci + oh, h - sqh)
    sqlocj = randint(0, w - sqw)
    crns = corners(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, cornc, crns)
    gomini = subgrid(oplcd, gi)
    goo = vupscale(hupscale(gomini, wscf), hscf)
    goo = asobject(goo)
    gloci = randint(sqloci + 1, sqloci + sqh - 1 - height(goo))
    glocj = randint(sqlocj + 1, sqlocj + sqw - 1 - width(goo))
    gooplcd = shift(goo, (gloci, glocj))
    go = paint(gi, gooplcd)
    go = subgrid(crns, go)
    indic = sfilter(gooplcd, lambda cij: cij[0] in (ac1, ac2))
    gi = paint(gi, indic)
    if choice((True, False)) and len(obj) > 3:
        idx = choice(totuple(toindices(sfilter(obj, lambda cij: cij[0] == objc))))
        idxi, idxj = idx
        xx = shift(asindices(canvas(-1, (hscf, wscf))), (gloci + idxi * hscf, glocj + idxj * wscf))
        gi = fill(gi, objc, xx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    # Reading (from the golf entry, differing from the cell-component maker):
    # the small key picture outside the frame is ONE unit.  Its enlarged copy
    # is fitted as a whole -- scale AND placement chosen so that every fragment
    # already inside the frame agrees with it -- and then stamped colour layer
    # by colour layer (all blocks of one key colour at once), after which the
    # frame is cropped out as the answer.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the demonstration inputs are painted on
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    if examples:
        tally = Counter()
        for e in examples:
            g = np.asarray(e[0], dtype=int)
            tally[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
        cand = tally.most_common(1)[0][0]
        if np.any(I == cand) and cand == bgc:
            bgc = cand

    bycol = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                bycol.setdefault(v, []).append((r, c))

    # --- the frame: four lone pixels of one colour on the corners of a box ---
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts})
        cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if rs[1] - rs[0] < 2 or cs[1] - cs[0] < 2:
            continue
        inner = {int(I[r, c]) for r in range(rs[0] + 1, rs[1])
                 for c in range(cs[0] + 1, cs[1])} - {bgc, col}
        if len(inner) < 2:
            continue
        area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
        if best is None or area > best[0]:
            best = (area, col, rs[0], rs[1], cs[0], cs[1])
    _, cornc, r0, r1, c0, c1 = best

    # --- the key picture: everything non-background outside the frame -------
    key = {}
    for col, pts in bycol.items():
        for (r, c) in pts:
            if not (r0 <= r <= r1 and c0 <= c <= c1):
                key[(r, c)] = col
    kr0 = min(r for r, _ in key)
    kc0 = min(c for _, c in key)
    keyloc = {(r - kr0, c - kc0): v for (r, c), v in key.items()}
    kh = max(i for i, _ in keyloc) + 1
    kw = max(j for _, j in keyloc) + 1

    # --- fragments of the enlarged copy already inside the frame -------------
    inside = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            v = int(I[r, c])
            if v != bgc:
                inside[(r, c)] = v

    # --- fit the whole enlarged key against every fragment -------------------
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    fit = None
    pr, pc = min(inside)                 # one fragment cell, any will do
    pv = inside[(pr, pc)]
    for hs in range(1, ih // kh + 1):
        for ws in range(1, iw // kw + 1):
            for (i, j), v in sorted(keyloc.items()):
                if v != pv:
                    continue
                for di in range(hs):
                    for dj in range(ws):
                        org_r = pr - i * hs - di
                        org_c = pc - j * ws - dj
                        if org_r <= r0 or org_c <= c0:
                            continue
                        if org_r + kh * hs > r1 or org_c + kw * ws > c1:
                            continue
                        if all(keyloc.get(((r - org_r) // hs, (c - org_c) // ws)) == v2
                               for (r, c), v2 in inside.items()):
                            fit = (hs, ws, org_r, org_c)
                            break
                    if fit:
                        break
                if fit:
                    break
            if fit:
                break
        if fit:
            break
    hs, ws, org_r, org_c = fit

    # --- stamp the enlarged key, one colour layer at a time ------------------
    # layer = every hs x ws block that key colour occupies; cells that already
    # hold that colour in the current grid (the fragments) are left alone.
    grid = I.copy()
    ops, sels = [], []
    order = sorted(set(keyloc.values()),
                   key=lambda v: min(p for p, u in keyloc.items() if u == v))
    for col in order:
        cells = []
        for (i, j), v in sorted(keyloc.items()):
            if v != col:
                continue
            for r in range(org_r + i * hs, org_r + (i + 1) * hs):
                for c in range(org_c + j * ws, org_c + (j + 1) * ws):
                    if grid[r, c] != col:
                        cells.append((r, c))
        if cells:
            ops.append(col); sels.append(sel_of(cells))
            for r, c in cells:
                grid[r, c] = col

    # --- the frame is the answer canvas: crop to its corners (full rectangle)
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    ops.append(34); sels.append([0, 0, r1 - r0, c1 - c0])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8a004b2b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8a004b2b"
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
                                f"for task 8a004b2b"
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
                    f"Failed to build a complete episode for task 8a004b2b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8a004b2b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
