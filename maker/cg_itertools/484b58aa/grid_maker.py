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
    return {"noisec": random.choice(range(10))}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    return random.randint(lo, max(lo, hi))


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, **kw) -> dict:
    if noisec is None:
        noisec = random.choice(range(10))
    h = _unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = _unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    hp = _unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    wp = _unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
    remcols = [c for c in range(10) if c != noisec]
    numc = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, numc)
    tile = [[random.choice(ccols) for _ in range(wp)] for _ in range(hp)]
    go = [[-1] * w for _ in range(h)]
    locs = set()
    ofs = random.randint(1, hp - 1)
    for a in range(2 * (h // hp + 1)):
        for b in range(w // wp + 1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            for i in range(hp):
                for j in range(wp):
                    r, c = loci + i, locj + j
                    if 0 <= r < h and 0 <= c < w:
                        go[r][c] = tile[i][j]
    if any(v == -1 for row in go for v in row):
        raise ValueError("uncovered")
    full = [(li, lj) for (li, lj) in locs if li >= 0 and lj >= 0 and li + hp <= h and lj + wp <= w]
    numpatches = _unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = [row[:] for row in go]
    succ = tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = random.randint(2, 6); pw = random.randint(2, 6)
        if ph > h or pw > w:
            continue
        li = random.randint(0, h - ph); lj = random.randint(0, w - pw)
        gi2 = [row[:] for row in gi]
        for r in range(li, li + ph):
            for c in range(lj, lj + pw):
                gi2[r][c] = noisec
        ok = any(all(gi2[a + i][b + j] != noisec for i in range(hp) for j in range(wp)) for a, b in full)
        if ok:
            rows_ok = sum(1 for row in gi2 if noisec not in row)
            cols_ok = sum(1 for c in range(w) if all(gi2[r][c] != noisec for r in range(h)))
            if rows_ok >= 2 and cols_ok >= 2:
                succ += 1
                gi = gi2
    gi = np.array(gi); go = np.array(go)
    k = random.choice((0, 1, 2, 3))
    gi = np.rot90(gi, k); go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading: the wallpaper is repaired by PROPAGATION. Each noise patch is
    # peeled strip by strip (a row-run or column-run of still-hidden cells);
    # each strip is copied from the CURRENT grid at a lattice offset whose
    # cells are already known -- either intact in I or repaired by an earlier
    # strip -- re-reading the pattern along a line and using what has just
    # been filled in as context for the next line.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            cand &= set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
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
    clean = ~noise
    if not noise.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    def overlap(dr, dc):
        return max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)

    # translation lattice of the wallpaper, measured on intact cells of I
    lattice = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = overlap(dr, dc)
            if r1 <= r0 or c1 <= c0:
                continue
            m = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                continue
            A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if (A[m] == B[m]).all():
                lattice.append((abs(dr) + abs(dc), dr, dc))
    lattice.sort()
    lattice = [(dr, dc) for _, dr, dc in lattice[:200]]

    # noise patches = connected blobs of the noise colour in I
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if noise[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; comp = []
                while q:
                    x, y = q.popleft(); comp.append((x, y))
                    for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                        if 0 <= nx < h and 0 <= ny < w and noise[nx, ny] and not seen[nx, ny]:
                            seen[nx, ny] = True; q.append((nx, ny))
                comps.append(comp)

    hidden = noise.copy()          # cells still showing the noise colour
    cur = I.copy()

    def runs(comp):
        cs = set(p for p in comp if hidden[p])
        out = []
        for (r, c) in cs:
            if (r, c - 1) not in cs:       # horizontal run start
                L = 0
                while (r, c + L) in cs:
                    L += 1
                out.append((r, c, 0, 1, L))
            if (r - 1, c) not in cs:       # vertical run start
                L = 0
                while (r + L, c) in cs:
                    L += 1
                out.append((r, c, 1, 0, L))
        return out

    def source_ok(cells, dr, dc):
        for (r, c) in cells:
            a, b = r + dr, c + dc
            if not (0 <= a < h and 0 <= b < w) or hidden[a, b]:
                return False
        return True

    for comp in comps:             # finish one patch before the next
        while any(hidden[p] for p in comp):
            best = None
            for (r, c, vr, vc, L) in runs(comp):
                # whole run, or a piece of it anchored at either end
                for ln in range(L, 0, -1):
                    for start in ({0, L - ln}):
                        cells = [(r + vr * (start + k), c + vc * (start + k)) for k in range(ln)]
                        for vi, (dr, dc) in enumerate(lattice):
                            if source_ok(cells, dr, dc):
                                key = (-ln, vi, cells[0])
                                if best is None or key < best[0]:
                                    best = (key, cells, dr, dc)
                                break
                    if best is not None and best[0][0] == -ln:
                        break
            if best is None:
                break              # nothing recoverable left in this patch
            _, cells, dr, dc = best
            r0 = min(p[0] for p in cells); c0 = min(p[1] for p in cells)
            r1 = max(p[0] for p in cells); c1 = max(p[1] for p in cells)
            srcv = {(a, b): int(cur[a + dr, b + dc]) for (a, b) in cells}
            zc = [p for p in cells if srcv[p] == 0 and cur[p] != 0]
            if zc:                 # Paste is transparent for 0: write true 0s first
                ops.append(0); sels.append(sel_of(zc))
                for p in zc:
                    cur[p] = 0
            if any(srcv[p] != 0 for p in cells):
                # CopyO: the known strip one lattice step away (current grid)
                ops.append(29); sels.append([r0 + dr, c0 + dc, r1 - r0, c1 - c0])
                ops.append(30); sels.append([r0, c0, 0, 0])   # Paste onto the strip
                for p in cells:
                    if srcv[p] != 0:
                        cur[p] = srcv[p]
            for p in cells:
                hidden[p] = False

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
