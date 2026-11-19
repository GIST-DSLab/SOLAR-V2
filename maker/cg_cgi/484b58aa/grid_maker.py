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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = random.choice(cols)
    remcols = [c for c in cols if c != noisec]
    random.shuffle(remcols)
    return {"noisec": noisec, "ccols": remcols}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if noisec is None:
        noisec = choice(cols)
    if ccols is None:
        ccols = list(remove(noisec, cols))
        random.shuffle(ccols)
    ccols = [c for c in ccols if c != noisec]
    h = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (10, max(10, min(30, max_w))))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    numc = unifint(diff_lb, diff_ub, (2, min(9, len(ccols))))
    cc = ccols[:numc]
    pobj = frozenset({(choice(cc), ij) for ij in pinds})
    go = canvas(-1, (h, w))
    locs = set()
    ofs = randint(1, hp - 1)
    for a in range(2 * (h // hp + 1)):
        for b in range(w // wp + 1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    if any(v == -1 for row in go for v in row):
        raise ValueError("lattice left uncovered cells; retry")
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
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: the wallpaper's ROWS (or columns) repeat along one axis. A row
    # damaged by noise is the same row as an intact one a period away, so each
    # noise run in a line is repaired by copying the matching segment of a
    # same-pattern line from I and pasting it onto the run. Line by line.
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # --- noise colour: in every demo input, never in any demo output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            cand &= set(np.unique(np.asarray(ei)).tolist()) - set(np.unique(np.asarray(eo)).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:  # verifier heuristic: fewest components, then rarest
        def ncomp(col):
            m = (I == col); seen = np.zeros_like(m); n = 0
            for r in range(H):
                for c in range(W):
                    if m[r, c] and not seen[r, c]:
                        n += 1; st = [(r, c)]; seen[r, c] = True
                        while st:
                            x, y = st.pop()
                            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                                if 0 <= nx < H and 0 <= ny < W and m[nx, ny] and not seen[nx, ny]:
                                    seen[nx, ny] = True; st.append((nx, ny))
            return n
        cols = sorted(set(I.flatten().tolist()))
        nc = {c: ncomp(c) for c in cols}
        mn = min(nc.values())
        cnt = Counter(I.flatten().tolist())
        noisec = min((c for c in cols if nc[c] == mn), key=lambda c: cnt[c])

    ops, sels = [], []
    if not (I == noisec).any():
        ops.append(34); sels.append([0, 0, H - 1, W - 1])
        return ops, sels

    def valid_shift(G, dr, dc):
        h, w = G.shape
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r1 <= r0 or c1 <= c0:
            return False
        A = G[r0:r1, c0:c1]; B = G[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        m = (A != noisec) & (B != noisec)
        if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
            return False
        return bool((A[m] == B[m]).all())

    def axis_period(G):  # smallest pure shift along rows-axis (dr, 0)
        for p in range(1, G.shape[0]):
            if valid_shift(G, p, 0):
                return p
        return None

    # choose the line orientation: rows if a pure vertical period exists
    pv, ph = axis_period(I), axis_period(I.T)
    if pv is not None and (ph is None or pv <= ph):
        G, transposed = I, False
    elif ph is not None:
        G, transposed = I.T, True
    else:
        G, transposed = I, False
    h, w = G.shape

    # candidate source shifts: same line pattern a period away first
    vecs = []
    p_main = axis_period(G)
    if p_main:
        ks = sorted([k for k in range(-(h // p_main), h // p_main + 1) if k], key=abs)
        vecs += [(k * p_main, 0) for k in ks]
    p_oth = axis_period(G.T)
    if p_oth:
        ks = sorted([k for k in range(-(w // p_oth), w // p_oth + 1) if k], key=abs)
        vecs += [(0, k * p_oth) for k in ks]
    rest = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if (dr, dc) == (0, 0) or (dr, dc) in vecs:
                continue
            if valid_shift(G, dr, dc):
                rest.append((abs(dr) + abs(dc), dr, dc))
    vecs += [(dr, dc) for _, dr, dc in sorted(rest)]

    def tobox(r, c, n):  # frame row-segment -> actual bbox
        return [c, r, n - 1, 0] if transposed else [r, c, 0, n - 1]

    def tocell(r, c):
        return (c, r) if transposed else (r, c)

    clip = None                             # clipboard contents (frame orientation)
    cur = G.copy()
    fixed = (G != noisec)                   # cells holding their true colour

    def pasted(grid, arr, r, c):
        g = grid.copy()
        for k, v in enumerate(arr):
            if v != 0 and c + k < w:
                g[r, c + k] = v
        return g

    # pass 0 copies intact segments of the INPUT (CopyI); if some noise is out of
    # reach of any intact input copy, later passes copy from cells already
    # repaired on the working grid (CopyO).
    first = True
    while True:
        src = G if first else cur.copy()
        ok = fixed.copy()
        copy_op = 28 if first else 29
        progress = False
        for r in range(h):                  # one line at a time
            c = 0
            while c < w:
                if fixed[r, c]:
                    c += 1; continue
                e = c
                while e + 1 < w and not fixed[r, e + 1]:
                    e += 1
                pos = c                     # this noise run is [c, e]
                while pos <= e:
                    best = None
                    for dr, dc in vecs:
                        sr, sc = r + dr, pos + dc
                        L = 0
                        while (pos + L <= e and 0 <= sr < h and 0 <= sc + L < w
                               and ok[sr, sc + L]):
                            L += 1
                        if L > 0 and (best is None or L > best[0]):
                            best = (L, sr, sc)
                    if best is None:
                        pos += 1; continue
                    L, sr, sc = best
                    seg = src[sr, sc:sc + L].copy()
                    nz = [k for k in range(L) if seg[k] != 0]
                    if any(cur[r, pos + k] != seg[k] for k in nz):
                        new = pasted(cur, seg, r, pos)
                        reuse = False
                        if clip is not None:   # clipboard already carries this segment?
                            old = pasted(cur, clip, r, pos)
                            reuse = (all(old[r, pos + k] == seg[k] for k in nz) and
                                     all(old[r, x] == cur[r, x] or not fixed[r, x]
                                         for x in range(w) if not (pos <= x < pos + L)))
                            if reuse:
                                new = old
                        if not reuse:
                            ops.append(copy_op); sels.append(tobox(sr, sc, L))  # copy intact segment
                            clip = seg.copy()
                        ops.append(30); sels.append(tobox(r, pos, 1)[:2] + [0, 0])  # paste onto run
                        cur = new
                    zc = [k for k in range(L) if seg[k] == 0 and cur[r, pos + k] != 0]
                    if zc:                   # Paste can't write 0: paint those cells
                        ops.append(0); sels.append(sel_of([tocell(r, pos + k) for k in zc]))
                        for k in zc:
                            cur[r, pos + k] = 0
                    fixed[r, pos:pos + L] = True
                    progress = True
                    pos += L
                c = e + 1
        first = False
        if fixed.all() or not progress:
            break

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
