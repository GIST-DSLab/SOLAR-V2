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
from collections import Counter
from maker.sel_helpers import sel_of
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    remcols = [c for c in cols if c != noisec]
    numc = random.randint(2, 9)
    ccols = random.sample(remcols, numc)
    return {"noisec": noisec, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, noisec=None, ccols=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    if noisec is None:
        noisec = choice(cols)
    if ccols is None:
        remcols = remove(noisec, cols)
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (2, 9)))
    ccols = [c for c in ccols if c != noisec]
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
    if any(v == -1 for row in go for v in row):
        raise ValueError("lattice did not cover the canvas")
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: the wallpaper repeats with a pure period P along one axis, so every
    # line (row, or column) is identical to the line P away. A damaged line is
    # repaired stretch by stretch: each run of noise cells in it is covered by
    # copying the same stretch from an intact twin line (same period class first,
    # then a sheared lattice twin if the whole class is damaged there).
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- noise colour: present in every demo input, absent from every demo output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            cand &= set(np.unique(np.asarray(ei)).tolist()) - set(np.unique(np.asarray(eo)).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:
        cols = sorted(set(I.flatten().tolist()))
        def ncomp(col):
            m = (I == col); seen = np.zeros_like(m); n = 0
            for r in range(h):
                for c in range(w):
                    if m[r, c] and not seen[r, c]:
                        n += 1; st = [(r, c)]; seen[r, c] = True
                        while st:
                            x, y = st.pop()
                            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                                if 0 <= nx < h and 0 <= ny < w and m[nx, ny] and not seen[nx, ny]:
                                    seen[nx, ny] = True; st.append((nx, ny))
            return n
        nc = {c: ncomp(c) for c in cols}
        mn = min(nc.values())
        cnt = Counter(I.flatten().tolist())
        noisec = min((c for c in cols if nc[c] == mn), key=lambda c: cnt[c])

    if not (I == noisec).any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # --- line period: smallest P (down the rows or across the columns) such that
    # lines P apart agree on every intact cell.
    def period_ok(G, P):
        H = G.shape[0]
        if P >= H:
            return False
        for res in range(P):
            block = G[res::P]
            for col in block.T:
                if len(set(col.tolist()) - {noisec}) > 1:
                    return False
        return True

    best = None
    for axis in (0, 1):
        G = I if axis == 0 else I.T
        for P in range(2, G.shape[0]):
            if period_ok(G, P):
                if best is None or P < best[1]:
                    best = (axis, P)
                break
    if best is None:
        best = (0, h)
    axis, P = best
    G = I if axis == 0 else I.T          # frame where the period runs down the rows
    H, W = G.shape

    def to_bbox(y, c0, L):
        # stretch of frame line y, columns c0..c0+L-1 -> ARCLE bbox in I
        return [y, c0, 0, L - 1] if axis == 0 else [c0, y, L - 1, 0]

    def to_cell(y, c):
        return (y, c) if axis == 0 else (c, y)

    noise = (G == noisec)
    clean = ~noise
    # translations of the wallpaper: pure period multiples (k*P, 0) first, then
    # sheared lattice vectors that agree on all intact cells they overlap.
    pure = sorted(((k * P, 0) for k in range(-(H // P), H // P + 1) if k), key=lambda v: abs(v[0]))
    shear = []
    for dy in range(-H + 1, H):
        for dx in range(-W + 1, W):
            if dx == 0:
                continue
            r0, r1 = max(0, -dy), min(H, H - dy)
            c0, c1 = max(0, -dx), min(W, W - dx)
            if r1 <= r0 or c1 <= c0:
                continue
            A = G[r0:r1, c0:c1]; B = G[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
            m = clean[r0:r1, c0:c1] & clean[r0 + dy:r1 + dy, c0 + dx:c1 + dx]
            if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                continue
            if (A[m] == B[m]).all():
                shear.append((-int(m.sum()), dy, dx))   # best-supported first
    # a sheared vector plus any multiple of the pure period is still a translation
    ext = {}
    for rank, (_, dy, dx) in enumerate(sorted(shear)):
        for k in range(-(2 * H // P) - 1, 2 * H // P + 2):
            v = (dy + k * P, dx)
            if -H < v[0] < H:
                ext.setdefault(v, (rank, abs(v[0]) + abs(v[1])))
    shifts = pure + sorted(ext, key=lambda v: (ext[v], v))

    for y in range(H):                    # one damaged line at a time
        c = 0
        while c < W:
            if not noise[y, c]:
                c += 1
                continue
            e = c
            while e + 1 < W and noise[y, e + 1]:
                e += 1
            # noise run [c, e] on line y: cover it with intact stretches of twin lines
            s = c
            while s <= e:
                bestt = None
                for dy, dx in shifts:
                    yy = y + dy
                    if not (0 <= yy < H) or not (0 <= s + dx < W):
                        continue
                    L = 0
                    while s + L <= e and s + dx + L < W and not noise[yy, s + dx + L]:
                        L += 1
                    if L > 0 and (bestt is None or L > bestt[2]):
                        bestt = (yy, s + dx, L)
                    if bestt is not None and bestt[2] == e - s + 1:
                        break
                if bestt is None:            # no intact twin anywhere: leave the cell
                    s += 1
                    continue
                yy, sx, L = bestt
                src = G[yy, sx:sx + L]
                zc = [to_cell(y, s + k) for k in range(L) if src[k] == 0]
                if zc:                       # Paste is transparent for 0: blank those first
                    ops.append(0); sels.append(sel_of(zc))
                if (src != 0).any():
                    ops.append(28); sels.append(to_bbox(yy, sx, L))              # CopyI intact twin stretch
                    ops.append(30); sels.append(to_bbox(y, s, 0)[:2] + [0, 0])   # Paste onto the run
                s += L
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
