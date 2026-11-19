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


def _unifint(lb, ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return min(max(a, int(round(a + (b - a) * d))), b)


def _nbrs(cell):
    i, j = cell
    return {(i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj}


def sample_colors(num_examples=None) -> dict:
    bgc, fixc, mainc = random.sample(range(10), 3)
    return {"bgc": bgc, "fixc": fixc, "mainc": mainc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fixc=None, mainc=None) -> dict:
    if bgc is None or fixc is None or mainc is None:
        bgc, fixc, mainc = random.sample(range(10), 3)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = random.randint(2, 5)
    ow = random.randint(2, 5)
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    obj = {random.choice(sorted(bounds))}
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for _ in range(ncells - 1):
        opts = set()
        for c in obj:
            opts |= _nbrs(c)
        obj.add(random.choice(sorted((bounds - obj) & opts)))
    r0 = min(i for i, j in obj); c0 = min(j for i, j in obj)
    obj = {(i - r0, j - c0) for i, j in obj}
    oh = max(i for i, j in obj) + 1
    ow = max(j for i, j in obj) + 1
    fixp = random.choice(sorted(obj))
    remcols = [c for c in range(10) if c not in (bgc, fixc, mainc)]
    gi = [[bgc] * w for _ in range(h)]
    cobj = {(fixc, fixp)} | {(mainc, p) for p in obj if p != fixp}
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    plcd = {(v, (i + loci, j + locj)) for v, (i, j) in cobj}
    for v, (i, j) in plcd:
        gi[i][j] = v
    go = [row[:] for row in gi]
    near = set()
    for _, p in plcd:
        near |= _nbrs(p)
    inds = {(i, j) for i in range(h) for j in range(w) if gi[i][j] == bgc} - near
    nocc = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(cobj)))))
    tr = succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = random.randint(1, 4)
        ups = {(v, (i * fac + a, j * fac + b)) for v, (i, j) in cobj
               for a in range(fac) for b in range(fac)}
        hh, ww = oh * fac, ow * fac
        cands = [ij for ij in inds if ij[0] <= h - hh and ij[1] <= w - ww]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        up = {(v, (i + li, j + lj)) for v, (i, j) in ups}
        upi = {p for _, p in up}
        if not upi.issubset(inds):
            continue
        succ += 1
        newc = random.choice(remcols)
        fixp2i = {p for v, p in up if v == fixc}
        for p in upi:
            inds -= _nbrs(p)
        for i, j in fixp2i:
            gi[i][j] = fixc; go[i][j] = fixc
        remfull = upi - fixp2i
        ntorem = _unifint(diff_lb, diff_ub, (0, max(0, len(remfull) - 1)))
        ntokeep = len(remfull) - ntorem
        a0 = min(i for i, j in fixp2i) - 1; b0 = min(j for i, j in fixp2i) - 1
        a1 = max(i for i, j in fixp2i) + 1; b1 = max(j for i, j in fixp2i) + 1
        outbox = {(i, b) for i in range(a0, a1 + 1) for b in (b0, b1)} | \
                 {(a, j) for j in range(b0, b1 + 1) for a in (a0, a1)}
        tokeep = {random.choice(sorted(remfull & outbox))}
        for _ in range(ntokeep - 1):
            full = set()
            for p in tokeep | fixp2i:
                full |= _nbrs(p)
            tokeep.add(random.choice(sorted((remfull & full) - tokeep)))
        for i, j in tokeep:
            gi[i][j] = newc
        for i, j in remfull:
            go[i][j] = newc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared by the maker and the golf entry -- they agree on the unit):
       * the marker colour is the one colour every object carries;
       * the template is the object with a single marker cell (golf: one cell not of
         the body colour);
       * every other object is that template upscaled by fac, where fac*fac is the
         size of its marker block, anchored so the marker blocks coincide;
       * completing a copy = painting, in the copy's own colour, one fac x fac block
         per template pixel.  Both readings paint block-by-block; the golf entry
         only fills empty cells, which here is the same as skipping blocks that are
         already complete.  So the op unit is unchanged: one Color op per
         template pixel's block, grown outward from the marker.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: shared by every demonstration input -- the colour common to all
    # of them (and I) with the largest total area; I's majority colour as fallback.
    tot = Counter(I.flatten().tolist())
    common = set(tot)
    for ex in (examples or []):
        EI = np.asarray(ex[0], dtype=int)
        cnt = Counter(EI.flatten().tolist())
        common &= set(cnt)
        tot.update(cnt)
    bgc = max(common, key=lambda c: tot[c]) if common else tot.most_common(1)[0][0]

    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                seen[r, c] = True
                q = deque([(r, c)]); cells = []
                while q:
                    y, x = q.popleft(); cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] != bgc:
                                seen[ny, nx] = True; q.append((ny, nx))
                objs.append(cells)
    if len(objs) < 2:
        return [34], [[0, 0, hi - 1, wi - 1]]

    pres = Counter()
    for cells in objs:
        for col in {int(I[y, x]) for y, x in cells}:
            pres[col] += 1
    fixc = max(pres.items(), key=lambda kv: kv[1])[0]
    nfix = lambda cells: sum(1 for y, x in cells if I[y, x] == fixc)
    mn = min(nfix(cs) for cs in objs)
    template = max([cs for cs in objs if nfix(cs) == mn], key=len)
    tr0 = min(y for y, x in template); tc0 = min(x for y, x in template)
    tnorm = {(y - tr0, x - tc0) for y, x in template}
    fixcell = next((y - tr0, x - tc0) for y, x in template if I[y, x] == fixc)

    order = []
    vis = {fixcell}; q = deque([fixcell])
    while q:
        y, x = q.popleft()
        if (y, x) != fixcell:
            order.append((y, x))
        for n in sorted((y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx):
            if n in tnorm and n not in vis:
                vis.add(n); q.append(n)

    G = I.copy()
    ops, sels = [], []
    copies = sorted([cs for cs in objs if cs is not template], key=min)
    for cells in copies:
        fixcells = [(y, x) for y, x in cells if I[y, x] == fixc]
        body = {int(I[y, x]) for y, x in cells if I[y, x] != fixc}
        if not fixcells or not body:
            continue
        newc = body.pop()
        fr = min(y for y, x in fixcells); fc = min(x for y, x in fixcells)
        fac = int(round(len(fixcells) ** 0.5))          # marker block is fac x fac
        off_r = fr - fixcell[0] * fac; off_c = fc - fixcell[1] * fac
        for (y, x) in order:
            br, bc = off_r + y * fac, off_c + x * fac
            r0, c0 = max(br, 0), max(bc, 0)
            r1, c1 = min(br + fac, hi), min(bc + fac, wi)
            if r1 <= r0 or c1 <= c0:
                continue
            if np.all(G[r0:r1, c0:c1] == newc):
                continue                                 # block already complete
            # full rectangle: this template pixel's fac x fac block in the copy
            ops.append(newc); sels.append([r0, c0, r1 - r0 - 1, c1 - c0 - 1])
            G[r0:r1, c0:c1] = newc
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
