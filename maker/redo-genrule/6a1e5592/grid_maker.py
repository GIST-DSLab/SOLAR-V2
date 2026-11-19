"""
ARC Task: 6a1e5592 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    while True:
        barc, bgc, objc = random.sample(cols, 3)
        if objc != 0:
            break
    n_ex = num_examples if num_examples else 3
    rots = [0, 1, 2, 3]
    if n_ex >= len(rots):
        examples = [{"rot": r} for r in rots]
        examples += [{"rot": random.choice(rots)} for _ in range(n_ex - len(rots))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(rots, n_ex)]
    plan = [dict(e) for e in examples]
    plan.append(dict(random.choice(examples)))
    return {"barc": barc, "bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, barc, bgc, objc, rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    def _uni(bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = int(np.ceil(a + (b - a) * float(diff_lb)))
        hi = int(np.floor(a + (b - a) * float(diff_ub)))
        lo = max(a, min(b, lo)); hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    def _shifted(mask, di, dj):
        hh, ww = mask.shape
        out = np.zeros_like(mask)
        r0 = max(0, -di); r1 = min(hh, hh - di)
        c0 = max(0, -dj); c1 = min(ww, ww - dj)
        if r0 < r1 and c0 < c1:
            out[r0:r1, c0:c1] = mask[r0 + di:r1 + di, c0 + dj:c1 + dj]
        return out

    def _dil(mask):
        out = np.zeros_like(mask)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                out |= _shifted(mask, di, dj)
        return out

    def _norm(s):
        a = min(i for i, j in s); b = min(j for i, j in s)
        return frozenset(((i - a, j - b) for i, j in s))

    if rot % 2 == 1:
        hlim, wlim = int(max_w), int(max_h)
    else:
        hlim, wlim = int(max_h), int(max_w)
    hi_h = max(6, min(30, hlim)); lo_h = min(9, hi_h)
    hi_w = max(3, min(30, wlim)); lo_w = min(5, hi_w)

    def _finish(grid, go):
        gi_r = np.ascontiguousarray(np.rot90(grid, k=rot)) if rot else grid
        go_r = np.ascontiguousarray(np.rot90(go, k=rot)) if rot else go
        if gi_r.shape[0] > int(max_h) or gi_r.shape[1] > int(max_w):
            return None
        mc = int(np.bincount(gi_r.flatten(), minlength=10).argmax())
        cnt = 0
        for t in (gi_r, gi_r.T, np.rot90(gi_r, 2).T, np.flipud(gi_r)):
            row0 = t[0]
            if len(set(row0.tolist())) == 1 and int(row0[0]) != mc:
                cnt += 1
        if cnt != 1:
            return None
        return {"input": gi_r.tolist(), "output": go_r.tolist()}

    # ---- verifier reimplementation: valid landing spots of a normalised patch ----
    def _valid_locs(cells, bgm, barbg, cand_mask):
        m = cand_mask.copy()
        for (i, j) in cells:
            m &= _shifted(bgm, i, j)
            if not m.any():
                return m
        cs = set(cells)
        halo = set()
        for (i, j) in cells:
            for (di, dj) in ((-1, 0), (0, -1), (0, 1)):
                p = (i + di, j + dj)
                if p not in cs:
                    halo.add(p)
        for (i, j) in halo:
            m &= ~_shifted(barbg, i, j)
            if not m.any():
                return m
        return m

    for _attempt in range(600):
        h = _uni((lo_h, hi_h)); w = _uni((lo_w, hi_w))
        if h < 6 or w < 3:
            continue
        bh_hi = max(2, h // 3)
        barh = random.randint(min(3, bh_hi), bh_hi)
        maxobjh = h - barh - 1
        if barh < 2 or maxobjh < 1:
            continue
        nobjs = _uni((1, max(1, w // 3)))

        grid = np.full((h, w), bgc, dtype=int)
        grid[:barh, :] = barc
        placopts = set(range(1, w - 1))
        if not placopts:
            continue
        free = set((r, c) for r in range(barh, h) for c in range(w))
        forb, srcs_all, dests_all, placements = set(), set(), set(), []
        tr, succ, maxtr = 0, 0, 10 * nobjs

        while tr < maxtr and succ < nobjs and placopts:
            tr += 1
            oh = random.randint(1, maxobjh)
            ow = random.randint(1, max(1, min(4, w // 2)))
            ncells = random.randint(1, oh * ow)
            cells = set([(0, random.randint(0, ow - 1))])
            for _ in range(ncells - 1):
                grow = sorted(set((i + di, j + dj) for (i, j) in cells
                                  for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1))
                                  if 0 <= i + di < oh and 0 <= j + dj < ow
                                  and (i + di, j + dj) not in cells))
                if not grow:
                    break
                cells.add(random.choice(grow))
            cells = _norm(cells)
            oh2 = max(i for i, j in cells) + 1
            ow2 = max(j for i, j in cells) + 1
            markerh = random.randint(1, min(oh2, barh - 1))
            mpn = _norm(set((i, j) for (i, j) in cells if i < markerh))
            prefixes = [_norm(set((i, j) for (i, j) in mpn if i < k))
                        for k in range(1, markerh + 1)]
            if any(p in forb for p in prefixes):
                continue
            jcands = [j for j in sorted(placopts) if set(range(j, j + ow2 + 1)) <= placopts]
            if not jcands:
                continue
            jloc = random.choice(jcands)
            iloc = barh - markerh
            dest = set((i + iloc, j + jloc) for (i, j) in cells)
            if any(r >= h or c >= w for (r, c) in dest):
                continue
            if (dest & srcs_all) or (dest & dests_all):
                continue
            icands = [(r, c) for (r, c) in free if r + oh2 <= h and c + ow2 <= w]
            if not icands:
                continue
            random.shuffle(icands)
            src = None; srcloc = None
            for loc in icands[:80]:
                trial = set((i + loc[0], j + loc[1]) for (i, j) in cells)
                if not trial <= free:
                    continue
                if (trial & dest) or (trial & dests_all):
                    continue
                src, srcloc = trial, loc
                break
            if src is None:
                continue
            for (r, c) in src:
                grid[r, c] = objc
            for (r, c) in dest:
                if r < barh:
                    grid[r, c] = bgc                      # carve the marker notch
            srcs_all |= src
            dests_all |= dest
            free -= src
            free -= set((r + di, c + dj) for (r, c) in src
                        for di in (-1, 0, 1) for dj in (-1, 0, 1))
            free -= dest
            for p in prefixes:
                forb.add(p)
            jm = set(c for (r, c) in dest)
            placopts -= (jm | set(c - 1 for c in jm) | set(c + 1 for c in jm))
            placements.append((cells, srcloc, (iloc, jloc)))
            succ += 1

        if succ == 0:
            continue

        vals, cnts = np.unique(grid, return_counts=True)
        order = np.argsort(-cnts)
        if int(vals[order[0]]) != bgc:
            continue
        if len(cnts) > 1 and int(cnts[order[0]]) == int(cnts[order[1]]):
            continue
        if set(int(v) for v in vals.tolist()) != {barc, bgc, objc}:
            continue

        # every object must land on its own notch, as the unique lowest-row legal spot
        bgm = (grid == bgc)
        cand_mask = _dil(_dil(bgm))
        barbg = bgm.copy(); barbg[barh:, :] = False
        ok = True
        for (cells, srcloc, destloc) in placements:
            m = _valid_locs(cells, bgm, barbg, cand_mask)
            rs, cs2 = np.nonzero(m)
            if len(rs) == 0:
                ok = False
                break
            mn = int(rs.min())
            tied = [(int(a), int(b)) for a, b in zip(rs, cs2) if int(a) == mn]
            if len(tied) != 1 or tied[0] != destloc:
                ok = False
                break
        if not ok:
            continue

        go = grid.copy()
        for (cells, srcloc, destloc) in placements:
            for (i, j) in cells:
                go[srcloc[0] + i, srcloc[1] + j] = bgc
        for (cells, srcloc, destloc) in placements:
            for (i, j) in cells:
                go[destloc[0] + i, destloc[1] + j] = 1

        res = _finish(grid, go)
        if res is not None:
            return res

    # deterministic fallback: one single-cell object and its one-cell notch
    h = hi_h; w = hi_w
    barh = max(2, min(3, max(2, h // 3)))
    grid = np.full((h, w), bgc, dtype=int)
    grid[:barh, :] = barc
    grid[barh - 1, 1] = bgc
    grid[h - 1, w - 1] = objc
    go = grid.copy()
    go[h - 1, w - 1] = bgc
    go[barh - 1, 1] = 1
    res = _finish(grid, go)
    if res is None:
        res = {"input": grid.tolist(), "output": go.tolist()}
    return res


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    vals, cnts = np.unique(I, return_counts=True)
    bgc = int(vals[int(np.argmax(cnts))])          # background = mostcolor, as the task defines it

    # the shapes that travel: non-background in I, background in O
    src_cells = set((r, c) for r in range(h) for c in range(w)
                    if int(I[r, c]) != bgc and int(O[r, c]) == bgc)
    dst_cells = set((r, c) for r in range(h) for c in range(w) if int(O[r, c]) == 1)

    def comps(cellset):
        rem = set(cellset); out = []
        while rem:
            seed = min(rem)
            rem.discard(seed)
            cur = {seed}; stack = [seed]
            while stack:
                r, c = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        p = (r + dr, c + dc)
                        if p in rem:
                            rem.discard(p); cur.add(p); stack.append(p)
            out.append(cur)
        return out

    def norm(s):
        a = min(r for r, c in s); b = min(c for r, c in s)
        return frozenset(((r - a, c - b) for r, c in s)), (a, b)

    srcs = comps(src_cells)
    dsts = comps(dst_cells)
    used = [False] * len(srcs)
    pairs = []
    for d in dsts:
        dn, dloc = norm(d)
        for k, s in enumerate(srcs):
            if used[k]:
                continue
            sn, sloc = norm(s)
            if sn == dn:
                used[k] = True
                pairs.append((s, sloc, d, dloc))
                break
    pairs.sort(key=lambda t: (t[3][0], t[3][1]))

    for (s, sloc, d, dloc) in pairs:
        dr = dloc[0] - sloc[0]
        dc = dloc[1] - sloc[1]
        cur = sorted(s)
        grabbed = False
        if dr != 0:
            op = 20 if dr < 0 else 21
            step = -1 if dr < 0 else 1
            for _ in range(abs(dr)):
                ops.append(op)
                sels.append(sel_of([]) if grabbed else sel_of(cur))
                grabbed = True
                cur = [(r + step, c) for r, c in cur]
        if dc != 0:
            op = 23 if dc < 0 else 22
            step = -1 if dc < 0 else 1
            for _ in range(abs(dc)):
                ops.append(op)
                sels.append(sel_of([]) if grabbed else sel_of(cur))
                grabbed = True
                cur = [(r, c + step) for r, c in cur]
        hole = sorted(set(s) - set(cur))           # only the vacated footprint reads 0
        if bgc != 0 and hole:
            ops.append(bgc)
            sels.append(sel_of(hole))
        ops.append(1)                              # mark the slotted shape
        sels.append(sel_of(sorted(d)))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])              # full-grid rectangle for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6a1e5592"
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
                                f"for task 6a1e5592"
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
                    f"Failed to build a complete episode for task 6a1e5592 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6a1e5592-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
