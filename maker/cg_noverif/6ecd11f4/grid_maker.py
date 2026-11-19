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
from random import choice, randint
from collections import Counter, deque
import numpy as np
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    ccols = [c for c in cols if c != bgc]
    random.shuffle(ccols)
    return {"bgc": bgc, "fgc": fgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])
    remcols = remove(bgc, cols)
    if ccols is None:
        ccols = list(remcols)
        random.shuffle(ccols)
    ccols = [c for c in ccols if c != bgc]
    h = unifint(diff_lb, diff_ub, (2, 7))
    w = unifint(diff_lb, diff_ub, (2, 7))
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = tuple(ccols[:ncols])
    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)
    maxhscf = (2 * h + h + 1) // h
    maxwscf = (2 * w + w + 1) // w
    hscf = unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    lo_h, lo_w = hscf * h + h + 1, wscf * w + w + 1
    if lo_h > max_h or lo_w > max_w:
        raise ValueError("grid too small")
    fullh = unifint(diff_lb, diff_ub, (lo_h, max_h))
    fullw = unifint(diff_lb, diff_ub, (lo_w, max_w))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    for _ in range(1000):
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        gix = paint(gi, shift(obj, (loci, locj)))
        ofc = ofcolor(gix, fgc)
        delt = (fullinds - ofc)
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(delt2, lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow)
        if len(scands) == 0:
            continue
        locc = choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break
    else:
        raise ValueError("placement failed")
    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): the mono-colour blob is an
    (hs, ws) upscale of a small h*w stencil; sampling the blob every hs rows /
    ws cols gives that stencil, and it masks the mixed-colour h*w patch cell by
    cell. The golf entry carves it the same way (stride-sample the blob, AND it
    against the patch), so the route is the same: erase the patch cells under
    stencil holes (one Color op per connected hole region, selected as that
    region's own cells), then crop the canvas onto the patch.
    O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def components(bg):
        seen = np.zeros((hi, wi), dtype=bool)
        out = []
        for r in range(hi):
            for c in range(wi):
                if I[r, c] == bg or seen[r, c]:
                    continue
                seen[r, c] = True
                dq = deque([(r, c)])
                cells = [(r, c)]
                while dq:
                    y, x = dq.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bg:
                                seen[ny, nx] = True
                                cells.append((ny, nx))
                                dq.append((ny, nx))
                out.append(cells)
                if len(out) > 2:
                    return None
        return out

    def describe(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        pal = {int(I[r, c]) for r, c in cells}
        return (len(cells), r0, c0, max(rs) - r0 + 1, max(cs) - c0 + 1, pal, set(cells))

    def analyze(bg):
        comps = components(bg)
        if comps is None or len(comps) != 2:
            return None
        info = [describe(cl) for cl in comps]
        for a, b in ((0, 1), (1, 0)):
            n_p, pr, pc, ph, pw, ppal, _ = info[a]
            n_s, sr, sc, sh, sw, spal, blob = info[b]
            if len(spal) != 1 or len(ppal) < 2 or n_p != ph * pw:
                continue
            if sh % ph or sw % pw:
                continue
            hs, ws = sh // ph, sw // pw
            if hs < 2 or ws < 2:
                continue
            # blob must be an exact hs x ws upscale (membership in the blob itself,
            # since the patch may sit inside the blob's bbox)
            ok = all(((sr + i, sc + j) in blob) ==
                     ((sr + (i // hs) * hs, sc + (j // ws) * ws) in blob)
                     for i in range(sh) for j in range(sw))
            if ok:
                return (pr, pc, ph, pw), (sr, sc), hs, ws, blob
        return None

    found, bgc = None, None
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        found = analyze(int(cand))
        if found is not None:
            bgc = int(cand)
            break

    ops, sels = [], []
    if found is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    (pr, pc, ph, pw), (sr, sc), hs, ws, blob = found
    # stencil = blob sampled at stride (hs, ws); holes = stencil cells off the blob
    holes = {(i, j) for i in range(ph) for j in range(pw)
             if (sr + i * hs, sc + j * ws) not in blob}
    todo = set(holes)
    while todo:
        seed = min(todo)
        todo.discard(seed)
        region = {seed}
        dq = deque([seed])
        while dq:
            y, x = dq.popleft()
            for nb in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if nb in todo:
                    todo.discard(nb)
                    region.add(nb)
                    dq.append(nb)
        # erase this connected hole region of the patch to background
        ops.append(bgc)
        sels.append(sel_of([(pr + i, pc + j) for i, j in region]))
    # crop onto the whole patch rectangle (crop zero-fills, so 0-valued patch cells survive)
    ops.append(33); sels.append([pr, pc, ph - 1, pw - 1])
    ops.append(34); sels.append([0, 0, ph - 1, pw - 1])
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
