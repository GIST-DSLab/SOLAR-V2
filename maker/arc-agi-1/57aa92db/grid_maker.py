"""
ARC Task: 57aa92db (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # generator samples exactly three colors: bgc, fixc (marker), mainc (template body).
    # fixc is the role the whole rule hangs on -> must be identical across the episode.
    cols = list(range(10))
    bgc, fixc, mainc = random.sample(cols, 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, fixc: int, mainc: int) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    oh = randint(2, 5)
    ow = randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = choice(totuple(obj))
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // (4 * len(obj))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off I alone):
       * one colour (fixc) appears in EVERY object -> the marker colour.
       * the template = the object carrying only ONE marker cell and the most body cells.
       * every other object is a partially drawn copy of the template, upscaled by
         fac = width of its solid marker block, anchored so the marker blocks coincide.
         Complete each copy, in its own colour, growing outward from its marker.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- diagonally connected non-background objects ---
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                seen[r, c] = True
                q = deque([(r, c)])
                cells = []
                while q:
                    y, x = q.popleft()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                objs.append(cells)
    if not objs:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # --- marker colour: the colour present in the most objects ---
    pres = Counter()
    for cells in objs:
        for col in {int(I[y, x]) for y, x in cells}:
            pres[col] += 1
    fixc = max(pres.items(), key=lambda kv: kv[1])[0]

    def nfix(cells):
        return sum(1 for y, x in cells if I[y, x] == fixc)

    # --- template: single marker cell, largest body ---
    mn = min(nfix(cells) for cells in objs)
    cands = [cells for cells in objs if nfix(cells) == mn]
    template = max(cands, key=len)

    tr0 = min(y for y, x in template)
    tc0 = min(x for y, x in template)
    tnorm = {(y - tr0, x - tc0) for y, x in template}
    fixcell = next((y - tr0, x - tc0) for y, x in template if I[y, x] == fixc)

    # body cells ordered by growth outward from the marker
    order = []
    vis = {fixcell}
    q = deque([fixcell])
    while q:
        y, x = q.popleft()
        if (y, x) != fixcell:
            order.append((y, x))
        for n in sorted((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx):
            if n in tnorm and n not in vis:
                vis.add(n)
                q.append(n)

    ops, sels = [], []
    copies = [cells for cells in objs if cells is not template]
    copies.sort(key=lambda cells: min(cells))

    for cells in copies:
        fixcells = [(y, x) for y, x in cells if I[y, x] == fixc]
        body_cols = {int(I[y, x]) for y, x in cells if I[y, x] != fixc}
        if not fixcells or not body_cols:
            continue
        newc = body_cols.pop()
        fr = min(y for y, x in fixcells)
        fc = min(x for y, x in fixcells)
        fac = max(x for y, x in fixcells) - fc + 1        # upscale factor from marker block
        off_r = fr - fixcell[0] * fac
        off_c = fc - fixcell[1] * fac

        for (y, x) in order:                              # one template pixel -> one fac x fac block
            br, bc = off_r + y * fac, off_c + x * fac
            need = [(r, c) for r in range(br, br + fac) for c in range(bc, bc + fac)
                    if 0 <= r < hi and 0 <= c < wi and I[r, c] != newc]
            if not need:
                continue
            if len(need) == fac * fac:
                ops.append(newc)
                sels.append([br, bc, fac - 1, fac - 1])
            else:
                for (r, c) in need:
                    ops.append(newc)
                    sels.append([r, c, 0, 0])

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 57aa92db"
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
                                f"for task 57aa92db"
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
                    f"Failed to build a complete episode for task 57aa92db "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"57aa92db-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
