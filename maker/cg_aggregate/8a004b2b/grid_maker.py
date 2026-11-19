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
    bgc, cornc, ac1, ac2, objc = random.sample(range(10), 5)
    return {"bgc": bgc, "cornc": cornc, "ac1": ac1, "ac2": ac2, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cornc, ac1, ac2, objc) -> dict:
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
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # background: the colour every demonstration input is painted on
    cands = Counter(I.flatten().tolist())
    if examples:
        common = None
        for ei, _ in examples:
            s = {Counter(np.asarray(ei).flatten().tolist()).most_common(1)[0][0]}
            common = s if common is None else common & s
        bgc = next(iter(common)) if common and len(common) == 1 else cands.most_common(1)[0][0]
    else:
        bgc = cands.most_common(1)[0][0]

    bycol = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                bycol.setdefault(v, []).append((r, c))

    # frame: a colour forming exactly the 4 corners of a box that encloses markers
    best = None
    for col, pts in bycol.items():
        if len(pts) != 4:
            continue
        rs = sorted({p[0] for p in pts}); cs = sorted({p[1] for p in pts})
        if len(rs) != 2 or len(cs) != 2:
            continue
        if set(pts) != {(rs[0], cs[0]), (rs[0], cs[1]), (rs[1], cs[0]), (rs[1], cs[1])}:
            continue
        if rs[1] - rs[0] < 2 or cs[1] - cs[0] < 2:
            continue
        inner = {int(I[rr, cc]) for rr in range(rs[0] + 1, rs[1])
                 for cc in range(cs[0] + 1, cs[1])} - {bgc, col}
        if len(inner) < 2:
            continue
        area = (rs[1] - rs[0] + 1) * (cs[1] - cs[0] + 1)
        if best is None or area > best[0]:
            best = (area, rs[0], rs[1], cs[0], cs[1])
    _, r0, r1, c0, c1 = best

    # key pattern: every non-background cell outside the frame box
    key = {(r, c): v for v, pts in bycol.items() for (r, c) in pts
           if not (r0 <= r <= r1 and c0 <= c <= c1)}
    kr0 = min(r for r, _ in key); kc0 = min(c for _, c in key)
    inside = {(r, c): int(I[r, c]) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
              if I[r, c] != bgc}

    # anchor colours: inside/key count ratio equals the common block area
    kcnt = Counter(key.values()); icnt = Counter(inside.values())
    ratios = [icnt[c] / kcnt[c] for c in icnt if kcnt.get(c, 0) > 0]
    R = Counter(ratios).most_common(1)[0][0]
    anchors = {c for c in icnt if kcnt.get(c, 0) > 0 and icnt[c] / kcnt[c] == R}
    ia = [p for p, v in inside.items() if v in anchors]
    ka = [p for p, v in key.items() if v in anchors]
    hs = (max(r for r, _ in ia) - min(r for r, _ in ia) + 1) // \
         (max(r for r, _ in ka) - min(r for r, _ in ka) + 1)
    ws = (max(c for _, c in ia) - min(c for _, c in ia) + 1) // \
         (max(c for _, c in ka) - min(c for _, c in ka) + 1)
    org_r = min(r for r, _ in ia) - (min(r for r, _ in ka) - kr0) * hs
    org_c = min(c for _, c in ia) - (min(c for _, c in ka) - kc0) * ws

    ops, sels = [], []
    # 1. cut the canvas down to the corner frame (full rectangle, corners included)
    ops.append(33); sels.append([r0, c0, r1 - r0, c1 - c0])
    G = I[r0:r1 + 1, c0:c1 + 1].copy()

    # 2. stamp the key scaled by (hs, ws) as one unit, anchored on the marker blocks:
    #    one Color op per key colour, selecting every scaled block of that colour
    for col in sorted(set(key.values())):
        cells = []
        for (r, c), v in key.items():
            if v != col:
                continue
            br = org_r + (r - kr0) * hs - r0
            bc = org_c + (c - kc0) * ws - c0
            cells += [(br + a, bc + b) for a in range(hs) for b in range(ws)]
        if all(G[p] == col for p in cells):
            continue  # every block of this colour is already present (anchor markers)
        ops.append(col); sels.append(sel_of(cells))
        for p in cells:
            G[p] = col

    ho, wo = G.shape
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
