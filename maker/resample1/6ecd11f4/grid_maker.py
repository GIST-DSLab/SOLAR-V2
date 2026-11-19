"""
ARC Task: 6ecd11f4 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(list(range(10)), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, fgc=1, **kw) -> dict:
    cols = interval(0, 10, 1)
    h = _unifint(diff_lb, diff_ub, (2, 7))
    w = _unifint(diff_lb, diff_ub, (2, 7))
    remcols = remove(bgc, cols)
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(list(remcols), ncols)
    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = random.choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = random.choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = random.choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)
    maxhscf = (2 * h + h + 1) // h
    maxwscf = (2 * w + w + 1) // w
    hscf = _unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = _unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(random.choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((random.choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    lo_h, lo_w = hscf * h + h + 1, wscf * w + w + 1
    if lo_h > max_h or lo_w > max_w:
        raise ValueError("grid too small")
    fullh = _unifint(diff_lb, diff_ub, (lo_h, max_h))
    fullw = _unifint(diff_lb, diff_ub, (lo_w, max_w))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    for _ in range(2000):
        loci = random.randint(0, fullh - oh)
        locj = random.randint(0, fullw - ow)
        gix = paint(gi, shift(obj, (loci, locj)))
        ofc = ofcolor(gix, fgc)
        delt = fullinds - ofc
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(delt2, lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow)
        if len(scands) == 0:
            continue
        locc = random.choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break
    else:
        raise ValueError("placement failed")
    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {"input": gi, "output": go}


def _components(I, bgc):
    H, W = I.shape
    seen = np.zeros((H, W), bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or I[r, c] == bgc:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and I[ny, nx] != bgc:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append(cells)
    return comps


def _bgc_candidates(I):
    # background = the colour whose removal leaves exactly one single-coloured
    # stencil plus one multicoloured tile, the stencil an integer upscale of the tile
    out = []
    for bgc, _ in Counter(I.flatten().tolist()).most_common():
        comps = _components(I, bgc)
        if len(comps) != 2:
            continue
        nc = [len({int(I[r, c]) for r, c in cs}) for cs in comps]
        if sorted(nc)[0] != 1 or sorted(nc)[1] < 2:
            continue
        def dims(cs):
            rs = [r for r, _ in cs]; cs2 = [c for _, c in cs]
            return max(rs) - min(rs) + 1, max(cs2) - min(cs2) + 1
        tile = comps[int(np.argmax(nc))]; sten = comps[int(np.argmin(nc))]
        (th, tw), (sh, sw) = dims(tile), dims(sten)
        if len(tile) != th * tw or sh % th or sw % tw or sh // th < 2 or sw // tw < 2:
            continue
        out.append(bgc)
    return out


def _find_bgc(I, examples=None):
    cands = _bgc_candidates(I)
    if len(cands) > 1 and examples:
        for ex in examples:
            ec = _bgc_candidates(np.asarray(ex[0], dtype=int))
            keep = [c for c in cands if c in ec]
            if keep:
                cands = keep
    if cands:
        return cands[0]
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    bgc = _find_bgc(I, examples)
    comps = _components(I, bgc)
    ncol = lambda cs: len({int(I[r, c]) for r, c in cs})
    block = max(comps, key=ncol)       # the multicoloured tile
    shape_c = min(comps, key=ncol)     # the single-coloured upscaled stencil
    br0 = min(r for r, _ in block); bc0 = min(c for _, c in block)
    bh = max(r for r, _ in block) - br0 + 1
    bw = max(c for _, c in block) - bc0 + 1
    sr0 = min(r for r, _ in shape_c); sc0 = min(c for _, c in shape_c)
    sh = max(r for r, _ in shape_c) - sr0 + 1
    sw = max(c for _, c in shape_c) - sc0 + 1
    fy, fx = sh // bh, sw // bw
    shp = set(shape_c)
    # tile cells whose stencil cell (downscaled) is background
    hole = [(i, j) for i in range(bh) for j in range(bw)
            if (sr0 + i * fy, sc0 + j * fx) not in shp]
    ops, sels = [], []
    # crop canvas to the tile's full rectangle (the tile is a solid h x w block)
    ops.append(33); sels.append([br0, bc0, bh - 1, bw - 1])
    if hole:
        ops.append(bgc); sels.append(sel_of(hole))
    ops.append(34); sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6ecd11f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6ecd11f4"
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
                                f"for task 6ecd11f4"
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
                    f"Failed to build a complete episode for task 6ecd11f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6ecd11f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
