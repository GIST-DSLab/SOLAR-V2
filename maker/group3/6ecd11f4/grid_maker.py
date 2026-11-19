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
from collections import Counter, deque
from maker.sel_helpers import sel_of


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(max(a, round(a + (b - a) * d)), b)


def _nbrs(cell):
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    pool = [c for c in cols if c != bgc]
    return {"bgc": bgc, "fgc": fgc, "ccol_pool": pool}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, ccol_pool=None) -> dict:
    if ccol_pool is None:
        ccol_pool = [c for c in range(10) if c != bgc]
    h = _unifint(diff_lb, diff_ub, (2, 7))
    w = _unifint(diff_lb, diff_ub, (2, 7))
    ncols = _unifint(diff_lb, diff_ub, (2, min(9, len(ccol_pool))))
    ccols = random.sample(list(ccol_pool), ncols)
    inds = {(i, j) for i in range(h) for j in range(w)}
    nlocsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = random.choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = random.choice(sorted(inds))
    inds.discard(sp)
    shp = {sp}
    for _ in range(nlocs):
        cands = set()
        for x in shp:
            cands |= _nbrs(x)
        cands = (inds - shp) & cands
        shp.add(random.choice(sorted(cands)))
    r0 = min(r for r, _ in shp)
    c0 = min(c for _, c in shp)
    shp = {(r - r0, c - c0) for r, c in shp}
    h = max(r for r, _ in shp) + 1
    w = max(c for _, c in shp) + 1
    maxh = (3 * h + 1) // h
    maxw = (3 * w + 1) // w
    hscf = _unifint(diff_lb, diff_ub, (2, maxh))
    wscf = _unifint(diff_lb, diff_ub, (2, maxw))
    oh, ow = h * hscf, w * wscf
    objx = {(i, j): random.choice(ccols) for i in range(h) for j in range(w)}
    if len(set(objx.values())) == 1:
        k = random.choice(sorted(objx))
        objx[k] = random.choice([c for c in ccols if c != objx[k]])
    fullh = _unifint(diff_lb, diff_ub, (hscf * h + h + 1, max(hscf * h + h + 1, max_h)))
    fullw = _unifint(diff_lb, diff_ub, (wscf * w + w + 1, max(wscf * w + w + 1, max_w)))
    if fullh > max_h or fullw > max_w:
        raise ValueError("too big")
    full = {(i, j) for i in range(fullh) for j in range(fullw)}
    for _ in range(2000):
        li = random.randint(0, fullh - oh)
        lj = random.randint(0, fullw - ow)
        ofc = {(li + i, lj + j) for i in range(oh) for j in range(ow)
               if (i // hscf, j // wscf) in shp}
        nb = set()
        for x in ofc:
            nb |= _nbrs(x)
        delt2 = full - ofc - nb
        scands = [ij for ij in delt2 if ij[0] <= fullh - oh and ij[1] <= fullw - ow]
        if not scands:
            continue
        lc = random.choice(sorted(scands))
        pcells = {(lc[0] + i, lc[1] + j) for (i, j) in objx}
        if pcells <= delt2:
            gi = np.full((fullh, fullw), bgc, dtype=int)
            for (r, c) in ofc:
                gi[r, c] = fgc
            for (i, j), v in objx.items():
                gi[lc[0] + i, lc[1] + j] = v
            go = np.full((h, w), bgc, dtype=int)
            for (i, j), v in objx.items():
                if (i, j) in shp:
                    go[i, j] = v
            return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("placement failed")


def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): two objects on the background -- a solid multicolour
    h*w patch and a one-colour blob that is an exact (hs, ws) upscale of an
    h*w stencil. The answer is the patch with its stencil holes blanked to bg.
    Route (a different carving from erasing row-runs on the big canvas):
    first crop the canvas straight onto the patch rectangle, then blank ALL
    stencil holes in one stroke -- a single Color(bg) whose mask is the
    down-sampled blob's empty cells (the stencil's negative) laid onto the patch.
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
        return (len(cells), r0, c0, max(rs) - r0 + 1, max(cs) - c0 + 1,
                {int(I[r, c]) for r, c in cells}, set(cells))

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
        ops.append(34)
        sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    (pr, pc, ph, pw), (sr, sc), hs, ws, blob = found
    # 1. crop onto the patch -- selection is exactly the full patch rectangle
    #    (patch cells that are 0 stay 0: crop zero-fills)
    ops.append(33)
    sels.append([pr, pc, ph - 1, pw - 1])
    # 2. stencil negative (down-sampled blob's empty cells) in patch-local coords,
    #    blanked to background in one stroke
    holes = [(i, j) for i in range(ph) for j in range(pw)
             if (sr + i * hs, sc + j * ws) not in blob]
    if holes:
        ops.append(bgc)
        sels.append(sel_of(holes))
    ops.append(34)
    sels.append([0, 0, ph - 1, pw - 1])
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
