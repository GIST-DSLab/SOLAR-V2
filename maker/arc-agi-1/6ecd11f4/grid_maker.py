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
from collections import deque, Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, fgc: int) -> dict:
    cols = interval(0, 10, 1)
    hmax = min(7, (max_h - 1) // 3)
    wmax = min(7, (max_w - 1) // 3)
    if hmax < 2 or wmax < 2:
        raise ValueError('grid bounds too small for this task')
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
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
    maxhscf = min((2 * h + h + 1) // h, (max_h - h - 1) // h)
    maxwscf = min((2 * w + w + 1) // w, (max_w - w - 1) // w)
    if maxhscf < 2 or maxwscf < 2:
        raise ValueError('grid bounds too small for this task')
    hscf = unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    fullh = unifint(diff_lb, diff_ub, (hscf * h + h + 1, max_h))
    fullw = unifint(diff_lb, diff_ub, (wscf * w + w + 1, max_w))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    while True:
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        loc = (loci, locj)
        gix = paint(gi, shift(obj, loc))
        ofc = ofcolor(gix, fgc)
        delt = (fullinds - ofc)
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(
            delt2,
            lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow
        )
        if len(scands) == 0:
            continue
        locc = choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break
    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I, O is only used for its dimensions):
      I holds exactly two objects on a plain background:
        * a solid h*w rectangle of mixed colors  -> the 'patch'
        * a mono-color blob that is an exact (hs, ws) upscale of some h*w stencil
      Down-sampling the blob back to h*w gives a keep/erase stencil for the patch.
      Patch cells whose stencil cell is background get erased to background;
      then the patch itself is what gets submitted.
    Route: erase the stencil holes on the patch (while the blob is still in the
    grid, i.e. the stencil is derived before anything is destroyed), then crop
    the canvas down onto the patch.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
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
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        pal = {int(I[r, c]) for r, c in cells}
        return (len(cells), r0, c0, r1 - r0 + 1, c1 - c0 + 1, pal)

    def analyze(bg):
        comps = components(bg)
        if comps is None or len(comps) != 2:
            return None
        info = [describe(cl) for cl in comps]
        for a, b in ((0, 1), (1, 0)):
            n_p, pr, pc, ph, pw, ppal = info[a]
            n_s, sr, sc, sh, sw, spal = info[b]
            if len(spal) != 1 or len(ppal) < 2:
                continue
            if n_p != ph * pw:                      # patch must be a solid rectangle
                continue
            if sh % ph or sw % pw:
                continue
            hs, ws = sh // ph, sw // pw
            if hs < 2 or ws < 2:
                continue
            fg = next(iter(spal))
            ok = True
            for i in range(sh):                     # blob must be an exact hs x ws upscale
                for j in range(sw):
                    v = I[sr + i, sc + j] == fg
                    ref = I[sr + (i // hs) * hs, sc + (j // ws) * ws] == fg
                    if v != ref:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            return (pr, pc, ph, pw), (sr, sc), hs, ws, fg
        return None

    found = None
    bgc = None
    for cand, _ in Counter(I.flatten().tolist()).most_common():
        found = analyze(int(cand))
        if found is not None:
            bgc = int(cand)
            break

    ops, sels = [], []

    if found is not None:
        (pr, pc, ph, pw), (sr, sc), hs, ws, fg = found
        # stencil holes, expressed in patch-local coordinates
        holes = {(i, j) for i in range(ph) for j in range(pw)
                 if I[sr + i * hs, sc + j * ws] != fg}
        # erase hole regions one connected region at a time
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
            rs = [r for r, _ in region]
            cs = [c for _, c in region]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(region) == (r1 - r0 + 1) * (c1 - c0 + 1):
                ops.append(bgc)
                sels.append([pr + r0, pc + c0, r1 - r0, c1 - c0])
            else:
                for r in range(r0, r1 + 1):
                    row = sorted(c for (rr, c) in region if rr == r)
                    k = 0
                    while k < len(row):
                        m = k
                        while m + 1 < len(row) and row[m + 1] == row[m] + 1:
                            m += 1
                        ops.append(bgc)
                        sels.append([pr + r, pc + row[k], 0, row[m] - row[k]])
                        k = m + 1
        ops.append(33)
        sels.append([pr, pc, ph - 1, pw - 1])

    ho, wo = O.shape
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
