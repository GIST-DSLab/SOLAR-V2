"""
ARC Task: 447fd412 (RE-ARC) — LLM-generated grid_maker
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
from utils import *
from dsl import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, indic, mainc = random.sample(cols, 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def _plan(I):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # 8-connected, multicolour components of non-background cells
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]; seen[r, c] = True; cells = []
                while st:
                    a, b = st.pop(); cells.append((a, b))
                    for da in (-1, 0, 1):
                        for db in (-1, 0, 1):
                            x, y = a + da, b + db
                            if 0 <= x < h and 0 <= y < w and not seen[x, y] and I[x, y] != bgc:
                                seen[x, y] = True; st.append((x, y))
                comps.append(cells)
    multi = [cs for cs in comps if len({I[a, b] for a, b in cs}) >= 2]
    if not multi:
        return bgc, None, []
    # template = the two-coloured object; indicator = its minority colour
    tmpl = max(multi, key=len)
    cnt = Counter(int(I[a, b]) for a, b in tmpl)
    indic = min(cnt, key=lambda k: cnt[k])
    mainc = max(cnt, key=lambda k: cnt[k])
    r0 = min(a for a, b in tmpl); c0 = min(b for a, b in tmpl)
    ind_rel = [(a - r0, b - c0) for a, b in tmpl if I[a, b] == indic]
    main_rel = [(a - r0, b - c0) for a, b in tmpl if I[a, b] == mainc]
    oh = max(a for a, b in tmpl) - r0 + 1
    ow = max(b for a, b in tmpl) - c0 + 1
    tset = set(tmpl)
    comp_of = {}
    for k, cs in enumerate(comps):
        for p in cs:
            comp_of[p] = k
    placements = []
    for fac in range(1, 6):
        hh, ww = oh * fac, ow * fac
        ind_up = [(a * fac + i, b * fac + j) for a, b in ind_rel for i in range(fac) for j in range(fac)]
        main_up = [(a * fac + i, b * fac + j) for a, b in main_rel for i in range(fac) for j in range(fac)]
        for r in range(h - hh + 1):
            for c in range(w - ww + 1):
                ic = [(r + a, c + b) for a, b in ind_up]
                if any(p in tset or I[p] != indic for p in ic):
                    continue
                ks = {comp_of[p] for p in ic}
                if sum(len(comps[k]) for k in ks) != len(ic):
                    continue
                mc = [(r + a, c + b) for a, b in main_up]
                if any(I[p] != bgc for p in mc):
                    continue
                placements.append((fac, r, c, mc))
    return bgc, mainc, placements


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, mainc, placements = _plan(I)
    ops, sels = [], []
    # each indicator-only fragment: complete the template body around it,
    # scaled by the fragment's upscale factor, in the template's body colour
    for fac, r, c, mc in placements:
        ops.append(mainc); sels.append(sel_of(sorted(mc)))
    ops.append(34); sels.append([0, 0, h - 1, w - 1])
    return ops, sels


def _apply(I, ops, sels):
    G = np.array(I, dtype=int)
    for op, s in zip(ops[:-1], sels[:-1]):
        for (a, b) in s["cells"]:
            G[a, b] = op
    return G


def generate(diff_lb, diff_ub, max_h, max_w, bgc, indic, mainc) -> dict:
    gi = go = None
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (min(12, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(12, max_w), max_w))
        oh = unifint(diff_lb, diff_ub, (1, 4))
        ow = unifint(diff_lb, diff_ub, (1, 4))
        if oh * ow < 3:
            if choice((True, False)):
                oh = unifint(diff_lb, diff_ub, (3, 4))
            else:
                ow = unifint(diff_lb, diff_ub, (3, 4))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (3, oh * ow))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        objt = totuple(obj)
        kk = len(obj)
        nindic = randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
        indicobj = set(sample(objt, nindic))
        mainobj = obj - indicobj
        obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        go = paint(go, plcd)
        inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
        fullinds = asindices(gi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
        tr = 0
        maxtr = 5 * noccs
        succ = 0
        while succ < noccs and tr < maxtr:
            tr += 1
            fac = randint(1, max(1, min(5, min(h, w) // max(oh, ow) - 1)))
            outobj = upscale(obj, fac)
            inobj = sfilter(outobj, lambda cij: cij[0] == indic)
            hh, ww = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            inobjp = shift(inobj, loc)
            outobjp = shift(outobj, loc)
            outobjp = sfilter(outobjp, lambda cij: cij[1] in fullinds)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, toindices(inobjp))
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        if succ == 0:
            continue
        gia = np.array(gi, dtype=int)
        goa = np.array(go, dtype=int)
        if Counter(gia.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        try:
            ops, sels = derive_operations(gia)
            if not np.array_equal(_apply(gia, ops, sels), goa):
                continue
        except Exception:
            continue
        return {"input": gi, "output": go}
    return {"input": gi, "output": go}


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
                        f"num_examples+1 ({num_examples + 1}) for task 447fd412"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 447fd412"
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
                                f"for task 447fd412"
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
                    f"Failed to build a complete episode for task 447fd412 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"447fd412-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
