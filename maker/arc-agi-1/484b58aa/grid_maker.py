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
import numpy as np
from collections import deque, Counter


def sample_colors(num_examples=None) -> dict:
    # only the noise colour carries a rule role (it marks the damaged patches);
    # the pattern colours are irrelevant to the rule, so they stay per-instance.
    return {"noisec": random.choice(list(range(10)))}


def generate(diff_lb, diff_ub, max_h, max_w, noisec) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(-1, (h, w))
    locs = set()
    ofs = randint(1, hp - 1)
    for a in range(2 * (h // hp + 1)):
        for b in range(w // wp + 1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotopts = (identity, rot90, rot180, rot270) if (h <= max_w and w <= max_h) else (identity, rot180)
    rotf = choice(rotopts)
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # Rule: the grid is one wallpaper pattern repeated on a translation lattice;
    # solid patches of a single "noise" colour hide parts of it. Each damaged
    # region is restored by copying the lattice-equivalent intact block of I.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    diff = (I != O)
    if not diff.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    noisec = Counter(I[diff].tolist()).most_common(1)[0][0]
    noise = (I == noisec)
    clean = ~noise

    def overlap(dr, dc):
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        return r0, r1, c0, c1

    # the pattern's translation lattice, measured from I alone:
    # shifts under which every pair of intact cells agrees (with real support).
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
            if int(m.sum()) < max(20, (r1 - r0) * (c1 - c0) // 8):
                continue
            if (A[m] == B[m]).all():
                lattice.append((abs(dr) + abs(dc), dr, dc))
    lattice.sort()
    lattice = lattice[:120]

    # per lattice shift: where an intact source block is available
    masks = []
    for _, dr, dc in lattice:
        m = np.zeros((h, w), bool)
        r0, r1, c0, c1 = overlap(dr, dc)
        m[r0:r1, c0:c1] = clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        masks.append((dr, dc, m))

    # damaged regions as connected blobs
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if diff[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; comp = []
                while q:
                    x, y = q.popleft(); comp.append((x, y))
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and diff[nx, ny] and not seen[nx, ny]:
                                seen[nx, ny] = True; q.append((nx, ny))
                comps.append(sorted(comp))

    remaining = diff.copy()
    cur = I.copy()
    for comp in comps:                      # finish one blob before the next
        while True:
            todo = [p for p in comp if remaining[p[0], p[1]]]
            if not todo:
                break
            sr, sc = todo[0]
            ii = np.zeros((h + 1, w + 1), int)
            ii[1:, 1:] = np.cumsum(np.cumsum(remaining.astype(int), 0), 1)
            # largest block anchored here whose lattice-shifted source is intact
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
                    if run < maxw:
                        maxw = run
                    for ww in range(1, maxw + 1):
                        cov = ii[r1 + 1, sc + ww] - ii[sr, sc + ww] - ii[r1 + 1, sc] + ii[sr, sc]
                        key = (-cov, (r1 - sr + 1) * ww, abs(dr) + abs(dc))
                        if best is None or key < best[0]:
                            best = (key, dr, dc, r1 - sr, ww - 1)
            if best is None:
                ops.append(int(O[sr, sc])); sels.append([sr, sc, 0, 0])
                cur[sr, sc] = O[sr, sc]; remaining[sr, sc] = False
                continue
            _, dr, dc, dh, dw = best
            src = I[sr + dr:sr + dr + dh + 1, sc + dc:sc + dc + dw + 1]
            tgt = cur[sr:sr + dh + 1, sc:sc + dw + 1]
            # clear first only where the source carries 0 (Paste is transparent there)
            if ((src == 0) & (tgt != 0)).any():
                ops.append(0); sels.append([sr, sc, dh, dw])
                tgt[:] = 0
            if ((src != 0) & (tgt != src)).any():
                ops.append(28); sels.append([sr + dr, sc + dc, dh, dw])   # CopyI intact block
                ops.append(30); sels.append([sr, sc, 0, 0])               # Paste over damage
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
