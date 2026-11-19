"""
ARC Task: 484b58aa (RE-ARC) — LLM-generated grid_maker
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
    ROTS = ["identity", "rot90", "rot180", "rot270"]
    noisec = random.choice(list(range(10)))
    remcols = [c for c in range(10) if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"noisec": noisec, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, noisec, ccols, rot=None) -> dict:
    if rot is None:
        rot = random.choice(["identity", "rot90", "rot180", "rot270"])
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    lim_h, lim_w = max_h, max_w
    if rot in ("rot90", "rot270"):
        lim_h, lim_w = max_w, max_h
    lim_h = max(10, min(30, lim_h))
    lim_w = max(10, min(30, lim_w))
    ccols = list(ccols)
    while True:
        h = unifint(diff_lb, diff_ub, (10, lim_h))
        w = unifint(diff_lb, diff_ub, (10, lim_w))
        hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
        wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
        pinds = asindices(canvas(-1, (hp, wp)))
        pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
        go = canvas(-1, (h, w))
        locs = set()
        ofs = randint(1, hp - 1)
        for b in range(-1, w // wp + 2):
            for a in range(-2, (h + ofs * abs(b)) // hp + 3):
                loci = hp * a - ofs * b
                locj = wp * b
                if -hp < loci < h and -wp < locj < w:
                    locs.add((loci, locj))
                    go = paint(go, shift(pobj, (loci, locj)))
        if any(-1 in row for row in go):
            continue
        inb = [(li, lj) for (li, lj) in locs
               if 0 <= li and 0 <= lj and li + hp <= h and lj + wp <= w]
        if not inb:
            continue
        numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
        gi = tuple(e for e in go)
        succ = 0
        tr = 0
        maxtr = max(25, 5 * numpatches)
        while succ < numpatches and tr < maxtr:
            tr += 1
            ph = randint(2, 6)
            pw = randint(2, 6)
            loci = randint(0, h - ph)
            locj = randint(0, w - pw)
            ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
            gi2 = fill(gi, noisec, ptch)
            intact = any(all(gi2[li + di][lj + dj] != noisec
                             for di in range(hp) for dj in range(wp))
                         for (li, lj) in inb)
            if intact:
                if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
                   len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                    succ += 1
                    gi = gi2
        if succ >= 1:
            break
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the golf entry): the wallpaper repeats LINE BY LINE.
    # Row r equals row r+p (or column c equals column c+p) for a small period p,
    # so every damaged line is mended from another line of the same residue
    # class: each run of noise cells in the line is overwritten by the intact
    # stretch of a sibling line lying in the same columns (rows).
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- noise colour: present in every demo input, absent from every output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            cand &= set(np.unique(np.asarray(ei)).tolist()) - set(np.unique(np.asarray(eo)).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:
        def ncomp(col):
            m = (I == col); seen = np.zeros_like(m); n = 0
            for r in range(h):
                for c in range(w):
                    if m[r, c] and not seen[r, c]:
                        n += 1; q = [(r, c)]; seen[r, c] = True
                        while q:
                            x, y = q.pop()
                            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                                if 0 <= nx < h and 0 <= ny < w and m[nx, ny] and not seen[nx, ny]:
                                    seen[nx, ny] = True; q.append((nx, ny))
            return n
        cols = sorted(set(I.flatten().tolist()))
        nc = {c: ncomp(c) for c in cols}
        mn = min(nc.values())
        cnt = Counter(I.flatten().tolist())
        noisec = min((c for c in cols if nc[c] == mn), key=lambda c: cnt[c])

    noise = (I == noisec)
    if not noise.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # --- line period along an axis: smallest p with line i == line i+p on intact cells
    def line_period(G, N):
        n, m = G.shape
        for p in range(1, n):
            ok, overlap = True, 0
            for i in range(n - p):
                msk = ~N[i] & ~N[i + p]
                overlap += int(msk.sum())
                if (G[i][msk] != G[i + p][msk]).any():
                    ok = False; break
            if ok and overlap >= max(8, m):
                return p
        return None

    cur = I.copy()
    remaining = noise.copy()

    def mend(axis, p):
        # axis 0: lines are rows; axis 1: lines are columns.
        G = I if axis == 0 else I.T
        N = noise if axis == 0 else noise.T
        R = remaining if axis == 0 else remaining.T   # views
        C = cur if axis == 0 else cur.T
        n, m = G.shape
        for i in range(n):
            j = 0
            while j < m:
                if not R[i, j]:
                    j += 1; continue
                e = j
                while e + 1 < m and R[i, e + 1]:
                    e += 1
                # this run of noise cells [j, e] in line i; cover it from sibling lines
                s = j
                while s <= e:
                    best = None
                    for k in range(i % p, n, p):
                        if k == i:
                            continue
                        t = s
                        while t <= e and not N[k, t]:
                            t += 1
                        if t > s:
                            key = (-(t - s), abs(k - i))
                            if best is None or key < best[0]:
                                best = (key, k, t - 1)
                    if best is None:
                        s += 1          # no sibling intact here; leave for fallback
                        continue
                    _, k, t = best
                    src = G[k, s:t + 1]
                    tgt = C[i, s:t + 1]
                    zc = [b for b in range(len(src)) if src[b] == 0 and tgt[b] != 0]
                    if zc:   # Paste is transparent for 0: blank those cells first
                        cells = [(i, s + b) if axis == 0 else (s + b, i) for b in zc]
                        ops.append(0); sels.append(sel_of(cells))
                        for b in zc:
                            C[i, s + b] = 0
                    if ((src != 0) & (tgt != src)).any():
                        if axis == 0:
                            ops.append(28); sels.append([k, s, 0, t - s])   # CopyI sibling row stretch
                            ops.append(30); sels.append([i, s, 0, 0])       # Paste onto the damaged row
                        else:
                            ops.append(28); sels.append([s, k, t - s, 0])   # CopyI sibling column stretch
                            ops.append(30); sels.append([s, i, 0, 0])       # Paste onto the damaged column
                        nz = src != 0
                        tgt[nz] = src[nz]
                    R[i, s:t + 1] = False
                    s = t + 1
                j = e + 1

    pr = line_period(I, noise)
    pc = line_period(I.T, noise.T)
    order = []
    if pr is not None and pr < h:
        order.append((pr, 0))
    if pc is not None and pc < w:
        order.append((pc, 1))
    order.sort()
    for p, axis in order:
        if remaining.any():
            mend(axis, p)

    # --- fallback: cells whose whole row/column class is hidden: use a 2-D
    # lattice shift of the wallpaper measured on intact cells
    if remaining.any():
        clean = ~noise
        def overlap(dr, dc):
            return max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
        lattice = []
        for dr in range(-h + 1, h):
            for dc in range(-w + 1, w):
                if dr == 0 and dc == 0:
                    continue
                r0, r1, c0, c1 = overlap(dr, dc)
                if r1 <= r0 or c1 <= c0:
                    continue
                A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                mm = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                if int(mm.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                    continue
                if (A[mm] == B[mm]).all():
                    lattice.append((abs(dr) + abs(dc), dr, dc))
        # close the lattice under addition: sums of measured shifts are shifts too
        vecs = {(dr, dc) for _, dr, dc in lattice}
        for _ in range(2):
            more = {(a + c, b + d) for (a, b) in vecs for (c, d) in vecs}
            for (a, b) in more:
                if (a, b) == (0, 0) or abs(a) >= h or abs(b) >= w or (a, b) in vecs:
                    continue
                r0, r1, c0, c1 = overlap(a, b)
                mm = clean[r0:r1, c0:c1] & clean[r0 + a:r1 + a, c0 + b:c1 + b]
                if (I[r0:r1, c0:c1][mm] == I[r0 + a:r1 + a, c0 + b:c1 + b][mm]).all():
                    vecs.add((a, b))
        lattice = sorted((abs(a) + abs(b), a, b) for (a, b) in vecs)
        for r in range(h):
            c = 0
            while c < w:
                if not remaining[r, c]:
                    c += 1; continue
                e = c
                while e + 1 < w and remaining[r, e + 1]:
                    e += 1
                s = c
                while s <= e:
                    best = None
                    for _, dr, dc in lattice:
                        rr = r + dr
                        if not (0 <= rr < h):
                            continue
                        t = s
                        while t <= e and 0 <= t + dc < w and clean[rr, t + dc]:
                            t += 1
                        if t > s and (best is None or t - s > best[0]):
                            best = (t - s, dr, dc, t - 1)
                    if best is None:
                        s += 1; continue
                    _, dr, dc, t = best
                    src = I[r + dr, s + dc:t + dc + 1]
                    tgt = cur[r, s:t + 1]
                    zc = [(r, s + b) for b in range(len(src)) if src[b] == 0 and tgt[b] != 0]
                    if zc:
                        ops.append(0); sels.append(sel_of(zc))
                        for (_, cc) in zc:
                            cur[r, cc] = 0
                    if ((src != 0) & (tgt != src)).any():
                        ops.append(28); sels.append([r + dr, s + dc, 0, t - s])
                        ops.append(30); sels.append([r, s, 0, 0])
                        nz = src != 0
                        tgt[nz] = src[nz]
                    remaining[r, s:t + 1] = False
                    s = t + 1
                c = e + 1

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 484b58aa"
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
                                f"for task 484b58aa"
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
                    f"Failed to build a complete episode for task 484b58aa "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"484b58aa-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
