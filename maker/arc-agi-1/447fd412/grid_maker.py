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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, indic, mainc = sample(cols, 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, indic: int, mainc: int) -> dict:
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
        fachi = min(5, min(h, w) // max(oh, ow) - 1)
        if fachi < 1:
            break
        fac = randint(1, fachi)
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
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (measured from I only):
      - one multicolour template object exists (indicator colour + body colour)
      - elsewhere, only the INDICATOR subset of that template appears, upscaled by
        some factor 1..5
      - each such occurrence is completed by painting the template's BODY cells
        (upscaled by the same factor) in the body colour
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- connected components of I (diagonal, background excluded) ---
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                                    and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                comps.append(cells)

    if not comps:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    # --- template = the component using the most distinct colours ---
    ti = max(range(len(comps)), key=lambda k: len({int(I[r, c]) for r, c in comps[k]}))
    tmpl = comps[ti]

    # --- indicator colour = the colour of the scattered (single-colour) pieces ---
    other_counts = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti
                           for (r, c) in cs)
    tmpl_counts = Counter(int(I[r, c]) for r, c in tmpl)
    if other_counts:
        indic = other_counts.most_common(1)[0][0]
    else:
        indic = min(tmpl_counts, key=lambda col: tmpl_counts[col])
    body_cols = [col for col in tmpl_counts if col != indic]
    if not body_cols:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    mainc = max(body_cols, key=lambda col: tmpl_counts[col])

    # --- normalized template ---
    tr0 = min(r for r, c in tmpl)
    tc0 = min(c for r, c in tmpl)
    norm = [(r - tr0, c - tc0, int(I[r, c])) for r, c in tmpl]
    ind_cells = [(a, b) for a, b, col in norm if col == indic]
    body_cells = [(a, b) for a, b, col in norm if col == mainc]
    all_cells = [(a, b) for a, b, col in norm]
    if not ind_cells or not body_cells:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    oh = max(a for a, b in all_cells) + 1
    ow = max(b for a, b in all_cells) + 1

    # --- padded canvas (input centred), so occurrences may hang off the edge ---
    H, W = 3 * h, 3 * w
    P = np.full((H, W), bgc, dtype=int)
    P[h:2 * h, w:2 * w] = I

    comp_of = {}
    for k, cs in enumerate(comps):
        for (r, c) in cs:
            comp_of[(r + h, c + w)] = k
    comp_size = [len(cs) for cs in comps]

    anchor = min(ind_cells)          # raster-first indicator cell of the template
    ind_positions = [(r + h, c + w) for r in range(h) for c in range(w)
                     if I[r, c] == indic]

    def fits(i, j, fac):
        return 0 <= i and 0 <= j and i + oh * fac <= H and j + ow * fac <= W

    def matches(i, j, fac):
        for (a, b) in ind_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != indic:
                        return False
        for (a, b) in body_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != bgc:
                        return False
        return True

    found = []
    seen_occ = set()
    for fac in range(1, 6):
        need = len(ind_cells) * fac * fac
        for (pr, pc) in ind_positions:
            i = pr - anchor[0] * fac
            j = pc - anchor[1] * fac
            if (fac, i, j) in seen_occ:
                continue
            seen_occ.add((fac, i, j))
            if not fits(i, j, fac) or not matches(i, j, fac):
                continue
            # the touched objects must be EXACTLY the upscaled indicator, nothing else
            hit = set()
            for (a, b) in all_cells:
                for da in range(fac):
                    for db in range(fac):
                        k = comp_of.get((i + a * fac + da, j + b * fac + db))
                        if k is not None:
                            hit.add(k)
            if sum(comp_size[k] for k in hit) != need:
                continue
            found.append((i, j, fac))

    # --- complete each occurrence: paint its body cells in the body colour ---
    found.sort(key=lambda t: (t[0], t[1], t[2]))
    for (i, j, fac) in found:
        paint_cells = []
        for (a, b) in body_cells:
            for da in range(fac):
                for db in range(fac):
                    r = i + a * fac + da - h
                    c = j + b * fac + db - w
                    if 0 <= r < h and 0 <= c < w:
                        paint_cells.append((r, c))
        if paint_cells:
            ops.append(int(mainc))
            sels.append(sel_of(sorted(set(paint_cells))))

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
