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
    """bgc / indic / mainc are the three roles the rule is stated in, so all
    three are fixed once per episode."""
    cols = list(range(10))
    bgc, indic, mainc = sample(cols, 3)
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, indic: int, mainc: int) -> dict:
    MAXD = 30
    hmax = min(max_h, 26)          # leave room for a stamp the frame cuts
    wmax = min(max_w, 26)
    for _attempt in range(60):
        h = unifint(diff_lb, diff_ub, (min(12, hmax), hmax))
        w = unifint(diff_lb, diff_ub, (min(12, wmax), wmax))
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
        fachi = min(5, min(h, w) // max(oh, ow) - 1)
        if fachi < 1:
            continue

        def place(inds, gi, go, over):
            """One stamp: the indicator part goes into the input, the whole
            upscaled template into the output.  `over` asks for a stamp the
            frame cuts — it may hang past the bottom/right edge as long as its
            indicator part stays inside and part of its body is cut off."""
            fac = randint(1, fachi)
            outobj = upscale(obj, fac)
            inobj = sfilter(outobj, lambda cij: cij[0] == indic)
            hh, ww = shape(outobj)
            ii = toindices(inobj)
            imr = max(i for i, j in ii)
            imc = max(j for i, j in ii)
            if over:
                cands = [ij for ij in inds
                         if ij[0] + imr < h and ij[1] + imc < w
                         and (ij[0] + hh > h or ij[1] + ww > w)
                         and h + max(0, ij[0] + hh - h) <= MAXD
                         and w + max(0, ij[1] + ww - w) <= MAXD]
            else:
                cands = [ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww]
            if not cands:
                return None
            loc = choice(cands)
            inobjp = shift(inobj, loc)
            outobjp = sfilter(shift(outobj, loc), lambda cij: cij[1] in fullinds)
            outobjpi = toindices(outobjp)
            if not outobjpi.issubset(inds):
                return None
            if over:
                body = [cij for cij in outobjp if cij[0] == mainc]
                if len(body) == 0 or \
                        len(body) == len(sfilter(outobj, lambda c: c[0] == mainc)):
                    return None      # nothing of the body left, or nothing cut
            inds = (inds - outobjpi) - mapply(neighbors, toindices(inobjp))
            return inds, paint(gi, inobjp), paint(go, outobjp)

        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
        tr, succ = 0, 0
        while succ < noccs and tr < 5 * noccs:
            tr += 1
            r = place(inds, gi, go, False)
            if r is not None:
                inds, gi, go = r
                succ += 1
        # at least one stamp the frame cuts: the rule draws the whole template
        # and the grid shows only the part inside its own borders
        cut = False
        for _ in range(12):
            r = place(inds, gi, go, True)
            if r is not None:
                inds, gi, go = r
                cut = True
                break
        if not cut or succ == 0:
            continue
        gia = np.array(gi, dtype=int)
        goa = np.array(go, dtype=int)
        if Counter(gia.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue                       # rule must be readable from I alone
        info = _analyse(gia)
        if info is None or not np.array_equal(_render(gia, info), goa):
            continue
        return {'input': gi, 'output': go}
    raise ValueError("447fd412: no instance with a cut stamp")


# ── reading the rule off the input ────────────────────────────────────────────

def _components(I, bgc):
    h, w = I.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                st = [(r, c)]
                seen[r, c] = True
                cells = []
                while st:
                    y, x = st.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                                    and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                st.append((ny, nx))
                comps.append(cells)
    return comps


def _analyse(I):
    """The template, and every stamp of it waiting to be completed.

    One component carries two colours: that is the template.  Every other
    component is its indicator subset, upscaled — so a stamp is a place where
    the upscaled indicator pattern sits with clear background wherever the
    template's body would go.  Stamps are searched on a canvas three times the
    grid, exactly as the task does, so a stamp whose body runs past an edge is
    found too, with its top-left given in the grid's own coordinates.
    """
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    comps = _components(I, bgc)
    if not comps:
        return None
    ti = max(range(len(comps)),
             key=lambda k: (len({int(I[r, c]) for r, c in comps[k]}), len(comps[k])))
    tmpl = comps[ti]
    tmpl_cols = {int(I[r, c]) for r, c in tmpl}
    if len(tmpl_cols) < 2:
        return None
    other = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti
                    for (r, c) in cs)
    if not other:
        return None
    indic = other.most_common(1)[0][0]
    body_cols = [c for c in tmpl_cols if c != indic]
    if len(body_cols) != 1:
        return None
    mainc = body_cols[0]
    tr0 = min(r for r, c in tmpl)
    tc0 = min(c for r, c in tmpl)
    norm = [(r - tr0, c - tc0, int(I[r, c])) for r, c in tmpl]
    ind_cells = sorted((a, b) for a, b, col in norm if col == indic)
    body_cells = sorted((a, b) for a, b, col in norm if col == mainc)
    if not ind_cells or not body_cells:
        return None
    oh = max(a for a, b, _ in norm) + 1
    ow = max(b for a, b, _ in norm) + 1

    H, W = 3 * h, 3 * w
    P = np.full((H, W), bgc, dtype=int)
    P[h:2 * h, w:2 * w] = I
    comp_of = {}
    for k, cs in enumerate(comps):
        for (r, c) in cs:
            comp_of[(r + h, c + w)] = k
    comp_size = [len(cs) for cs in comps]
    anchor = ind_cells[0]
    ind_pos = [(r + h, c + w) for r in range(h) for c in range(w) if I[r, c] == indic]

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

    placements, tried = [], set()
    for fac in range(1, 6):
        need = len(ind_cells) * fac * fac
        for (pr, pc) in ind_pos:
            i = pr - anchor[0] * fac
            j = pc - anchor[1] * fac
            if (fac, i, j) in tried:
                continue
            tried.add((fac, i, j))
            if i < 0 or j < 0 or i + oh * fac > H or j + ow * fac > W:
                continue
            if not matches(i, j, fac):
                continue
            hit = set()
            for (a, b) in ind_cells + body_cells:
                for da in range(fac):
                    for db in range(fac):
                        k = comp_of.get((i + a * fac + da, j + b * fac + db))
                        if k is not None:
                            hit.add(k)
            if sum(comp_size[k] for k in hit) != need:
                continue          # the indicator cells there are somebody else's
            placements.append((i - h, j - w, fac))
    placements.sort()
    return bgc, indic, mainc, oh, ow, ind_cells, body_cells, placements


def _body_at(r0, c0, fac, body_cells):
    return sorted({(r0 + a * fac + da, c0 + b * fac + db)
                   for (a, b) in body_cells
                   for da in range(fac) for db in range(fac)})


def _render(I, info):
    """The finished grid the rule asks for — used to validate generate()."""
    bgc, indic, mainc, oh, ow, ind_cells, body_cells, placements = info
    h, w = I.shape
    G = I.copy()
    for (r0, c0, fac) in placements:
        for (r, c) in _body_at(r0, c0, fac, body_cells):
            if 0 <= r < h and 0 <= c < w:
                G[r, c] = mainc
    return G


def derive_operations(I, O):
    """
    One two-coloured template, and elsewhere only its indicator subset, upscaled:
    each of those is a stamp of the template waiting for its body.  Some stamps
    are cut by the frame — their body runs past an edge — so the sheet is first
    ENLARGED to the size that holds every completed template, each template is
    drawn whole on it, and the sheet is cropped back to the original frame: the
    window the task looks through.  When nothing runs past an edge the sheet is
    already that window and neither resize is issued.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    info = _analyse(I)
    if info is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    bgc, indic, mainc, oh, ow, ind_cells, body_cells, placements = info

    MAXD = 30
    over_r = over_c = 0
    for (r0, c0, fac) in placements:            # how far the bodies reach past the frame
        for (a, b) in body_cells:
            over_r = max(over_r, r0 + (a + 1) * fac - h)
            over_c = max(over_c, c0 + (b + 1) * fac - w)
    dh = max(0, min(over_r, MAXD - h))
    dw = max(0, min(over_c, MAXD - w))
    H, W = h + dh, w + dw

    G = np.zeros((H, W), dtype=int)             # the grid as it stands after each op
    G[:h, :w] = I
    if dh or dw:
        # full rectangle: the whole enlarged sheet, the grid keeping its corner
        ops.append(33); sels.append([0, 0, H - 1, W - 1])

    for (r0, c0, fac) in placements:            # one op per stamp: its body, whole
        cells = [(r, c) for (r, c) in _body_at(r0, c0, fac, body_cells)
                 if 0 <= r < H and 0 <= c < W and G[r, c] != mainc]
        if not cells:
            continue
        ops.append(int(mainc)); sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = mainc

    if dh or dw:
        # back to the frame the task looks through
        ops.append(33); sels.append([0, 0, h - 1, w - 1])

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
