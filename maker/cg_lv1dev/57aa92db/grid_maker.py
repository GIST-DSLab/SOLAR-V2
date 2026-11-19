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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly three episode-level colours.
    # Each copy's own colour (newc) is still drawn per copy inside generate().
    bgc, fixc, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fixc=None, mainc=None, **kwargs) -> dict:
    if bgc is None or fixc is None or mainc is None:
        bgc, fixc, mainc = random.sample(range(10), 3)

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(max(a, round(a + (b - a) * d)), b)

    def nbrs(cell):
        i, j = cell
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj}

    def mnbrs(cells):
        out = set()
        for c in cells:
            out |= nbrs(c)
        return out

    def outbox(cells):
        rs = [i for i, j in cells]
        cs = [j for i, j in cells]
        r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
        box = set()
        for j in range(c0, c1 + 1):
            box.add((r0, j))
            box.add((r1, j))
        for i in range(r0, r1 + 1):
            box.add((i, c0))
            box.add((i, c1))
        return box

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = random.randint(2, 5)
    ow = random.randint(2, 5)
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    shape = {random.choice(sorted(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for _ in range(ncells - 1):
        opts = (bounds - shape) & mnbrs(shape)
        if not opts:
            break
        shape.add(random.choice(sorted(opts)))
    mr = min(i for i, j in shape)
    mc = min(j for i, j in shape)
    shape = {(i - mr, j - mc) for i, j in shape}
    oh = max(i for i, j in shape) + 1
    ow = max(j for i, j in shape) + 1
    fixp = random.choice(sorted(shape))
    remcols = [c for c in cols if c not in (bgc, fixc, mainc)]

    gi = [[bgc] * w for _ in range(h)]
    obj = {fixp: fixc}
    for c in shape:
        if c != fixp:
            obj[c] = mainc
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = {(i + loci, j + locj): v for (i, j), v in obj.items()}
    for (i, j), v in plcd.items():
        gi[i][j] = v
    go = [row[:] for row in gi]

    inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - mnbrs(set(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(obj)))))
    tr, succ, maxtr = 0, 0, 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        ups = {}
        for (i, j), v in obj.items():
            for a in range(fac):
                for b in range(fac):
                    ups[(i * fac + a, j * fac + b)] = v
        hh, ww = oh * fac, ow * fac
        cands = [ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        up = {(i + li, j + lj): v for (i, j), v in ups.items()}
        upi = set(up)
        if not upi.issubset(inds):
            continue
        succ += 1
        newc = random.choice(remcols)
        fix2 = {c for c, v in up.items() if v == fixc}
        inds = inds - mnbrs(upi)
        for (i, j) in fix2:
            gi[i][j] = fixc
            go[i][j] = fixc
        remfull = upi - fix2
        ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remfull) - 1)))
        ntokeep = len(remfull) - ntorem
        start = remfull & outbox(fix2)
        if not start:
            start = remfull
        tokeep = {random.choice(sorted(start))}
        for _ in range(ntokeep - 1):
            opts = (remfull & mnbrs(tokeep | fix2)) - tokeep
            if not opts:
                break
            tokeep.add(random.choice(sorted(opts)))
        for (i, j) in tokeep:
            gi[i][j] = newc
        for (i, j) in remfull:
            go[i][j] = newc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Golf reading: each copy is the template stamped once per marker pixel.
    The upscale factor is s = side of the solid marker block. Every marker
    pixel (f, u) stamps the template's body offsets (dr, dc) relative to its
    marker at stride s, landing on (f + s*dr, u + s*dc). The s*s strided
    stamps interleave to form the upscaled copy. So one Color op per marker
    pixel, in the copy's own body colour, selecting exactly where that stamp
    lands. That is named from I alone; O is never read.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # 8-connected non-background objects
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

    # marker colour = the colour present in every (the most) objects
    pres = Counter()
    for cells in objs:
        for col in {int(I[y, x]) for y, x in cells}:
            pres[col] += 1
    fixc = max(pres.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def markers(cells):
        return sorted((y, x) for y, x in cells if I[y, x] == fixc)

    # template = fewest marker cells (a single one), most body cells
    mn = min(len(markers(cells)) for cells in objs)
    cands = [cells for cells in objs if len(markers(cells)) == mn]
    template = max(cands, key=len)
    e, i = markers(template)[0]
    offsets = sorted((y - e, x - i) for y, x in template if I[y, x] != fixc)

    cur = I.copy()
    ops, sels = [], []
    copies = sorted((cells for cells in objs if cells is not template), key=lambda cells: min(cells))
    for cells in copies:
        mk = markers(cells)
        body_cols = Counter(int(I[y, x]) for y, x in cells if I[y, x] != fixc)
        if not mk or not body_cols:
            continue
        newc = body_cols.most_common(1)[0][0]
        s = max(x for y, x in mk) - min(x for y, x in mk) + 1   # marker block side = scale
        for (f, u) in mk:                                        # one strided stamp per marker pixel
            stamp = [(f + s * dr, u + s * dc) for dr, dc in offsets
                     if 0 <= f + s * dr < hi and 0 <= u + s * dc < wi]
            if not stamp or all(cur[r, c] == newc for r, c in stamp):
                continue                                         # this stamp is already fully drawn
            ops.append(newc)
            sels.append(sel_of(stamp))
            for r, c in stamp:
                cur[r, c] = newc

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
