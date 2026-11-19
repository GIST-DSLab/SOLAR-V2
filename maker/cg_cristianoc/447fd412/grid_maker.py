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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Background, indicator colour and body colour are all sampled by the generator.
    # Fix them for the whole episode so every instance uses the same colour roles.
    bgc, indic, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, indic=None, mainc=None,
             **kwargs) -> dict:
    if bgc is None or indic is None or mainc is None:
        bgc, indic, mainc = random.sample(range(10), 3)

    def unifint(bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(diff_lb, diff_ub)
        return min(max(a, int(round(a + (b - a) * d))), b)

    def nbrs(i, j):
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)
                if (di, dj) != (0, 0)}

    lo_h = min(12, max_h)
    lo_w = min(12, max_w)
    h = unifint((lo_h, max_h))
    w = unifint((lo_w, max_w))

    oh = unifint((1, 4))
    ow = unifint((1, 4))
    if oh * ow < 3:
        if random.choice((True, False)):
            oh = unifint((3, 4))
        else:
            ow = unifint((3, 4))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    ncells = unifint((3, oh * ow))
    obj = {random.choice(sorted(bounds))}
    for _ in range(ncells - 1):
        nb = set()
        for (i, j) in obj:
            nb |= nbrs(i, j)
        cands = sorted((bounds - obj) & nb)
        obj.add(random.choice(cands))
    mi = min(i for i, j in obj)
    mj = min(j for i, j in obj)
    obj = {(i - mi, j - mj) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    objt = sorted(obj)
    kk = len(obj)
    nindic = random.randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(random.sample(objt, nindic))
    mainobj = obj - indicobj
    cobj = [(indic, ij) for ij in indicobj] + [(mainc, ij) for ij in mainobj]

    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    plcd = [(v, (i + loci, j + locj)) for v, (i, j) in cobj]
    for v, (i, j) in plcd:
        gi[i][j] = v
        go[i][j] = v
    pn = set()
    for _, (i, j) in plcd:
        pn |= nbrs(i, j)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - pn

    noccs = unifint((1, max(1, (h * w) // (4 * len(plcd)))))
    tr, maxtr, succ = 0, 5 * noccs, 0
    while succ < noccs and tr < maxtr:
        tr += 1
        fac = random.randint(1, max(1, min(5, min(h, w) // max(oh, ow) - 1)))
        outobj = [(v, (i * fac + a, j * fac + b)) for v, (i, j) in cobj
                  for a in range(fac) for b in range(fac)]
        inobj = [(v, ij) for v, ij in outobj if v == indic]
        hh, ww = oh * fac, ow * fac
        cands = sorted(ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww)
        if not cands:
            continue
        li, lj = random.choice(cands)
        inobjp = [(v, (i + li, j + lj)) for v, (i, j) in inobj]
        outobjp = [(v, (i + li, j + lj)) for v, (i, j) in outobj
                   if 0 <= i + li < h and 0 <= j + lj < w]
        outobjpi = {ij for _, ij in outobjp}
        if outobjpi.issubset(inds):
            succ += 1
            nn = set()
            for _, (i, j) in inobjp:
                nn |= nbrs(i, j)
            inds = (inds - outobjpi) - nn
            for v, (i, j) in inobjp:
                gi[i][j] = v
            for v, (i, j) in outobjp:
                go[i][j] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    The code-golf solution and the maker read this task the same way:
      * the template is the component that uses the most colours;
      * the indicator is the template's minority colour (generator: nindic < |obj|/2),
        and it is also the colour of the scattered pieces;
      * elsewhere, a scaled copy (x1..x5) of just the template's indicator cells sits
        on its own;
      * each such copy is completed by stamping the scaled template there.
    Golf stamps the whole scaled template, indicator cells included. Those cells already
    hold the indicator colour, so painting them again would change nothing. That leaves
    one meaningful op per occurrence, the same as the maker's: paint the occurrence's
    scaled body cells (the stamp's footprint) in the body colour, clipped to the grid.
    Every value here is measured from I alone.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected components (background excluded)
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

    def finish():
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    if not comps:
        return finish()

    ti = max(range(len(comps)), key=lambda k: len({int(I[r, c]) for r, c in comps[k]}))
    tmpl = comps[ti]
    other_counts = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti
                           for (r, c) in cs)
    tmpl_counts = Counter(int(I[r, c]) for r, c in tmpl)
    if other_counts:
        indic = other_counts.most_common(1)[0][0]
    else:
        indic = min(tmpl_counts, key=lambda col: tmpl_counts[col])
    body_cols = [col for col in tmpl_counts if col != indic]
    if not body_cols:
        return finish()
    mainc = max(body_cols, key=lambda col: tmpl_counts[col])

    tr0 = min(r for r, c in tmpl)
    tc0 = min(c for r, c in tmpl)
    norm = [(r - tr0, c - tc0, int(I[r, c])) for r, c in tmpl]
    ind_cells = [(a, b) for a, b, col in norm if col == indic]
    body_cells = [(a, b) for a, b, col in norm if col == mainc]
    all_cells = [(a, b) for a, b, col in norm]
    if not ind_cells or not body_cells:
        return finish()
    oh = max(a for a, b in all_cells) + 1
    ow = max(b for a, b in all_cells) + 1

    # padded canvas so occurrences may hang off the edge
    H, W = 3 * h, 3 * w
    P = np.full((H, W), bgc, dtype=int)
    P[h:2 * h, w:2 * w] = I
    comp_of = {}
    for k, cs in enumerate(comps):
        for (r, c) in cs:
            comp_of[(r + h, c + w)] = k
    comp_size = [len(cs) for cs in comps]

    anchor = min(ind_cells)
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

    # stamp each occurrence: its scaled body cells, in the body colour
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

    return finish()


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
