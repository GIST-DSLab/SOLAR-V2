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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    bgc, fgc = random.sample(range(10), 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kwargs) -> dict:
    remcols = [c for c in range(10) if c != bgc]
    h = _unifint(diff_lb, diff_ub, (2, 7))
    w = _unifint(diff_lb, diff_ub, (2, 7))
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)
    inds = {(i, j) for i in range(h) for j in range(w)}
    nlocsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = random.choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = random.choice(sorted(inds))
    shp = {sp}
    for _ in range(nlocs):
        cands = set()
        for (a, b) in shp:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if (da or db) and (a + da, b + db) in inds and (a + da, b + db) not in shp:
                        cands.add((a + da, b + db))
        if not cands:
            break
        shp.add(random.choice(sorted(cands)))
    r0 = min(a for a, _ in shp); c0 = min(b for _, b in shp)
    shp = {(a - r0, b - c0) for a, b in shp}
    h = max(a for a, _ in shp) + 1
    w = max(b for _, b in shp) + 1
    hscf = _unifint(diff_lb, diff_ub, (2, (3 * h + 1) // h))
    wscf = _unifint(diff_lb, diff_ub, (2, (3 * w + 1) // w))
    oh, ow = h * hscf, w * wscf
    blob = {(a * hscf + x, b * wscf + y) for a, b in shp for x in range(hscf) for y in range(wscf)}
    patch = {(i, j): random.choice(ccols) for i in range(h) for j in range(w)}
    if len(set(patch.values())) == 1:
        k = next(iter(patch))
        patch[k] = random.choice([c for c in ccols if c != patch[k]])
    lo_h, lo_w = hscf * h + h + 1, wscf * w + w + 1
    if lo_h > max_h or lo_w > max_w:
        raise ValueError("grid too small")
    fullh = _unifint(diff_lb, diff_ub, (lo_h, max_h))
    fullw = _unifint(diff_lb, diff_ub, (lo_w, max_w))
    for _ in range(500):
        li = random.randint(0, fullh - oh)
        lj = random.randint(0, fullw - ow)
        ofc = {(a + li, b + lj) for a, b in blob}
        near = set(ofc)
        for (a, b) in ofc:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    near.add((a + da, b + db))
        scands = [(i, j) for i in range(fullh - oh + 1) for j in range(fullw - ow + 1)
                  if (i, j) not in near]
        if not scands:
            continue
        pi, pj = random.choice(scands)
        pcells = {(pi + a, pj + b) for a, b in patch}
        if all(0 <= a < fullh and 0 <= b < fullw and (a, b) not in near for a, b in pcells):
            gi = np.full((fullh, fullw), bgc, dtype=int)
            for a, b in ofc:
                gi[a, b] = fgc
            for (a, b), v in patch.items():
                gi[pi + a, pj + b] = v
            go = np.full((h, w), bgc, dtype=int)
            for (a, b), v in patch.items():
                if (a, b) in shp:
                    go[a, b] = v
            return {"input": gi.tolist(), "output": go.tolist()}
    raise ValueError("placement failed")


def derive_operations(I, O, examples=None):
    """
    Reading (shared by the maker and the golf entry): two objects on a plain
    background -- a solid multi-colour h*w 'patch' and a mono-colour blob that
    is an exact (hs, ws) upscale of an h*w stencil. The golf entry carves it
    as "lift the patch out first, then knock out the cells whose down-sampled
    stencil cell is empty"; that order is followed here:
      1. CropGrid onto the patch rectangle (the patch becomes the canvas),
      2. erase each connected stencil-hole region of the cropped patch to the
         background colour, one Color op per region (mask = that region's cells).
    The stencil is read from the blob's own cells in I, never from O.
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
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        pal = {int(I[r, c]) for r, c in cells}
        return (len(cells), r0, c0, r1 - r0 + 1, c1 - c0 + 1, pal, set(cells))

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
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    (pr, pc, ph, pw), (sr, sc), hs, ws, blob = found
    # 1. lift the patch out: crop onto the patch rectangle
    #    (bbox is exactly the patch, which is a solid rectangle)
    ops.append(33); sels.append([pr, pc, ph - 1, pw - 1])
    # 2. stencil holes, in patch-local == cropped-canvas coordinates
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
        ops.append(bgc); sels.append(sel_of(region))   # erase this hole region to background
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
