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
    import random
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
    import random
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
        for a in range(2 * (h // hp + 1)):
            for b in range(w // wp + 1):
                loci = hp * a - ofs * b
                locj = wp * b
                locs.add((loci, locj))
                go = paint(go, shift(pobj, (loci, locj)))
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
    # Rule: the grid is one wallpaper tile repeated on a translation lattice;
    # solid patches of a single noise colour hide parts of it. Every noise patch
    # is repaired by copying a lattice-equivalent, fully intact block of I onto it.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- noise colour: from the demonstrations (present in every input, absent
    # from every output); fallback: verifier heuristic on I alone.
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

    # --- translation lattice of the wallpaper, measured on intact cells of I
    lattice = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = overlap(dr, dc)
            if r1 <= r0 or c1 <= c0:
                continue
            A = I[r0:r1, c0:c1]
            B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            m = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                continue
            if (A[m] == B[m]).all():
                lattice.append((abs(dr) + abs(dc), dr, dc))
    lattice.sort()
    lattice = lattice[:160]

    masks = []   # where a lattice-shifted intact source cell exists
    for _, dr, dc in lattice:
        m = np.zeros((h, w), bool)
        r0, r1, c0, c1 = overlap(dr, dc)
        m[r0:r1, c0:c1] = clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        masks.append((dr, dc, m))

    # --- noise patches as connected blobs (seen in I)
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
                comps.append(sorted(comp))

    remaining = noise.copy()
    cur = I.copy()
    for comp in comps:                      # finish one patch before the next
        while True:
            todo = [p for p in comp if remaining[p[0], p[1]]]
            if not todo:
                break
            sr, sc = todo[0]
            ii = np.zeros((h + 1, w + 1), int)
            ii[1:, 1:] = np.cumsum(np.cumsum(remaining.astype(int), 0), 1)
            best = None
            for dr, dc, m in masks:
                if not m[sr, sc]:
                    continue
                maxw = w - sc
                for r1 in range(sr, h):
                    if not m[r1, sc]:
                        break
                    run = 0
                    while sc + run < w and m[r1, sc + run]:
                        run += 1
                    maxw = min(maxw, run)
                    for ww in range(1, maxw + 1):
                        cov = ii[r1 + 1, sc + ww] - ii[sr, sc + ww] - ii[r1 + 1, sc] + ii[sr, sc]
                        key = (-cov, (r1 - sr + 1) * ww, abs(dr) + abs(dc))
                        if best is None or key < best[0]:
                            best = (key, dr, dc, r1 - sr, ww - 1)
            if best is None:
                remaining[sr, sc] = False   # no intact lattice copy visible
                continue
            _, dr, dc, dh, dw = best
            src = I[sr + dr:sr + dr + dh + 1, sc + dc:sc + dc + dw + 1]
            tgt = cur[sr:sr + dh + 1, sc:sc + dw + 1]
            # Paste is transparent for 0: blank the noise cells whose true colour is 0
            zc = [(sr + a, sc + b) for a in range(dh + 1) for b in range(dw + 1)
                  if src[a, b] == 0 and tgt[a, b] != 0]
            if zc:
                ops.append(0); sels.append(sel_of(zc))
                for (a, b) in zc:
                    cur[a, b] = 0
            if ((src != 0) & (tgt != src)).any():
                ops.append(28); sels.append([sr + dr, sc + dc, dh, dw])   # CopyI intact block
                ops.append(30); sels.append([sr, sc, 0, 0])               # Paste over patch
                tgt[src != 0] = src[src != 0]
            remaining[sr:sr + dh + 1, sc:sc + dw + 1] = False

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
