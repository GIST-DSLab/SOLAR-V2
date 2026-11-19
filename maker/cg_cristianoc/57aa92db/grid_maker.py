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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc, fixc, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fixc=None, mainc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def neighbors(ij):
        i, j = ij
        return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj}

    def mapply_nb(cells):
        s = set()
        for c in cells:
            s |= neighbors(c)
        return s

    cols = list(range(10))
    if bgc is None or fixc is None or mainc is None:
        bgc, fixc, mainc = random.sample(cols, 3)
    remcols = [c for c in cols if c not in (bgc, fixc, mainc)]

    for _attempt in range(100):
        H = max(10, max_h)
        W = max(10, max_w)
        h = unifint(diff_lb, diff_ub, (10, H)) if max_h >= 10 else max_h
        w = unifint(diff_lb, diff_ub, (10, W)) if max_w >= 10 else max_w
        h = min(h, max_h)
        w = min(w, max_w)
        oh = random.randint(2, 5)
        ow = random.randint(2, 5)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(3, ncells), oh * ow)
        for _ in range(ncells - 1):
            opts = sorted((bounds - obj) & mapply_nb(obj))
            obj.add(random.choice(opts))
        mi = min(i for i, j in obj)
        mj = min(j for i, j in obj)
        obj = {(i - mi, j - mj) for i, j in obj}
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        if oh > h or ow > w:
            continue
        fixp = random.choice(sorted(obj))
        gi = [[bgc] * w for _ in range(h)]
        objc = {fixp: fixc}
        for p in obj:
            if p != fixp:
                objc[p] = mainc
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        plcd = {(i + loci, j + locj): c for (i, j), c in objc.items()}
        for (i, j), c in plcd.items():
            gi[i][j] = c
        go = [row[:] for row in gi]
        inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - mapply_nb(set(plcd))
        nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(obj)))))
        tr = 0
        succ = 0
        maxtr = 5 * nocc
        while succ < nocc and tr < maxtr:
            tr += 1
            fac = random.randint(1, 4)
            ups = {}
            for (i, j), c in objc.items():
                for a in range(fac):
                    for b in range(fac):
                        ups[(i * fac + a, j * fac + b)] = c
            hh, ww = oh * fac, ow * fac
            cands = [ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww]
            if not cands:
                continue
            loc = random.choice(sorted(cands))
            up = {(i + loc[0], j + loc[1]): c for (i, j), c in ups.items()}
            upi = set(up)
            if upi.issubset(inds):
                succ += 1
                newc = random.choice(remcols)
                fixp2 = {p for p, c in up.items() if c == fixc}
                inds = inds - mapply_nb(upi)
                for (i, j) in fixp2:
                    gi[i][j] = fixc
                    go[i][j] = fixc
                remfull = upi - fixp2
                ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remfull) - 1)))
                ntokeep = len(remfull) - ntorem
                r0 = min(i for i, j in fixp2) - 1
                r1 = max(i for i, j in fixp2) + 1
                c0 = min(j for i, j in fixp2) - 1
                c1 = max(j for i, j in fixp2) + 1
                outbox = {(i, j) for i in range(r0, r1 + 1) for j in range(c0, c1 + 1)
                          if i in (r0, r1) or j in (c0, c1)}
                tokeep = {random.choice(sorted(remfull & outbox))}
                for _ in range(ntokeep - 1):
                    opts = sorted((remfull & mapply_nb(tokeep | fixp2)) - tokeep)
                    if not opts:
                        break
                    tokeep.add(random.choice(opts))
                for (i, j) in tokeep:
                    gi[i][j] = newc
                for (i, j) in remfull:
                    go[i][j] = newc
        if succ == 0 or gi == go:
            continue
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    """Reading (shared with the code-golf solution): the marker colour sits in every
    object; the template is the object with one marker cell and the largest body.
    Each other object is a partial copy, one marker block of width fac.  The WHOLE
    template, upscaled by fac and anchored on that marker block, is stamped onto
    the copy in the copy's own colour as ONE unit: one Color op whose selection is
    'the cells the upscaled copy lands on' (minus the marker block, which stays)."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # diagonally connected multi-colour objects
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

    ops, sels = [], []
    if objs:
        pres = Counter()
        for cells in objs:
            for col in {int(I[y, x]) for y, x in cells}:
                pres[col] += 1
        fixc = max(pres.items(), key=lambda kv: (kv[1], -kv[0]))[0]

        def nfix(cells):
            return sum(1 for y, x in cells if I[y, x] == fixc)

        mn = min(nfix(c) for c in objs)
        template = max([c for c in objs if nfix(c) == mn], key=len)
        tr0 = min(y for y, x in template)
        tc0 = min(x for y, x in template)
        tnorm = sorted((y - tr0, x - tc0) for y, x in template)
        fixcell = next((y - tr0, x - tc0) for y, x in template if I[y, x] == fixc)

        copies = sorted([c for c in objs if c is not template], key=min)
        for cells in copies:
            fixcells = [(y, x) for y, x in cells if I[y, x] == fixc]
            body = sorted({int(I[y, x]) for y, x in cells if I[y, x] != fixc})
            if not fixcells or not body:
                continue
            newc = body[0]
            fr = min(y for y, x in fixcells)
            fc = min(x for y, x in fixcells)
            fac = max(x for y, x in fixcells) - fc + 1
            orr = fr - fixcell[0] * fac
            occ = fc - fixcell[1] * fac
            # cells the upscaled template lands on, anchored at this marker block
            land = [(orr + y * fac + a, occ + x * fac + b)
                    for (y, x) in tnorm if (y, x) != fixcell
                    for a in range(fac) for b in range(fac)
                    if 0 <= orr + y * fac + a < hi and 0 <= occ + x * fac + b < wi]
            # skip only if the stamp would change nothing on the current grid
            if any(I[r, c] != newc for r, c in land):
                ops.append(newc)
                sels.append(sel_of(land))

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
