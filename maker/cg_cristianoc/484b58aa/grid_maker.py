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
    # The rule hinges on the NOISE colour role (it is what gets repaired), so it is
    # fixed per episode; the wallpaper palette is fixed too so instances share it.
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
    if succ == 0:
        raise ValueError("no noise patch placed; retry")
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (from the golf entry): the wallpaper repeats along each AXIS on its
    # own -- every row has one horizontal period b, every column one vertical
    # period a -- so the picture is a single a x b RECTANGULAR tile laid on an
    # axis-aligned grid, and every cell's true colour is tile[r % a][c % b].
    # Noise cells are wildcards while the periods are measured.
    # Where the maker searches all (skewed) lattice translations at once, this
    # route is axis-first: a noise patch is repaired by CopyI of the same-sized,
    # fully intact block exactly (k*a, m*b) away -- whole tiles away -- and a
    # Paste over the patch. Each patch is done as one unit where an intact copy of
    # its whole bounding box exists.
    # Only if the rectangular repeat cannot fit in the grid (a heavily sheared
    # wallpaper whose row period is wider than the grid) does it fall back to the
    # sheared lattice. That lattice is built from its two shortest measured repeat
    # vectors.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- noise colour: present in every demo input, absent from every output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            cand &= set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:   # verifier heuristic: fewest components, then rarest
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

    # --- axis periods (noise = wildcard), as in the golf reading
    def row_period_ok(K):
        A, B = I[:, :-K], I[:, K:]
        m = clean[:, :-K] & clean[:, K:]
        return int(m.sum()) >= max(12, m.size // 8) and bool((A[m] == B[m]).all())

    def col_period_ok(K):
        A, B = I[:-K, :], I[K:, :]
        m = clean[:-K, :] & clean[K:, :]
        return int(m.sum()) >= max(12, m.size // 8) and bool((A[m] == B[m]).all())

    def tile_table(a, b):
        tab = {}
        for r in range(h):
            for c in range(w):
                if clean[r, c]:
                    k = (r % a, c % b)
                    if k in tab and tab[k] != I[r, c]:
                        return None
                    tab[k] = int(I[r, c])
        for r in range(h):
            for c in range(w):
                if noise[r, c] and (r % a, c % b) not in tab:
                    return None
        return tab

    a_cands = [K for K in range(1, h) if col_period_ok(K)]
    b_cands = [K for K in range(1, w) if row_period_ok(K)]
    found = None
    for a, b in sorted(((x, y) for x in a_cands for y in b_cands), key=lambda p: (p[0] * p[1], p)):
        tab = tile_table(a, b)
        if tab is not None:
            found = (a, b, tab)
            break
    if found is not None:
        a, b, tab = found
        truth = lambda r, c: tab[(r % a, c % b)]   # tile colour at a cell (read from I)
        shifts = [(ka * a, mb * b) for ka in range(-(h // a) - 1, h // a + 2)
                  for mb in range(-(w // b) - 1, w // b + 2) if (ka, mb) != (0, 0)]
    else:
        # The rectangular repeat does not fit in the grid (sheared wallpaper):
        # use the translation lattice under which intact cells agree.
        def consistent(dr, dc):
            r0, r1, c0, c1 = max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
            A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            m = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            return int(m.sum()), bool((A[m] == B[m]).all())

        def lattice(min_support):
            gens = []
            for dr in range(-h + 1, h):
                for dc in range(-w + 1, w):
                    if (dr, dc) <= (0, 0):
                        continue
                    sup, ok = consistent(dr, dc)
                    if ok and sup >= min_support(dr, dc):
                        gens.append((dr, dc))
            # The repeat vectors form a lattice: basis = shortest repeat plus the
            # next shortest one outside its span; close the span under +/-.
            gens.sort(key=lambda v: (abs(v[0]) + abs(v[1]), v))
            basis = []

            def span(bs):
                pts = set()
                R = 2 * (h + w)
                if len(bs) == 1:
                    (p, q), = bs
                    for i in range(-R, R + 1):
                        pts.add((i * p, i * q))
                    return pts
                fr = [(0, 0)]; pts.add((0, 0))
                while fr:
                    nxt = []
                    for u in fr:
                        for g in bs:
                            for v in ((u[0] + g[0], u[1] + g[1]), (u[0] - g[0], u[1] - g[1])):
                                if abs(v[0]) < 2 * h and abs(v[1]) < 2 * w and v not in pts:
                                    pts.add(v); nxt.append(v)
                    fr = nxt
                return pts

            span_pts = set()
            for g in gens:
                if len(basis) >= 2:
                    break
                if g not in span_pts:
                    basis.append(g)
                    span_pts = span(basis)
            found_v = span_pts - {(0, 0)}
            sh = [v for v in found_v if abs(v[0]) < h and abs(v[1]) < w and consistent(*v)[1]]
            sh.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s))
            tabd = {}
            for r in range(h):
                for c in range(w):
                    if noise[r, c]:
                        for dr, dc in sh:
                            if 0 <= r + dr < h and 0 <= c + dc < w and clean[r + dr, c + dc]:
                                tabd[(r, c)] = int(I[r + dr, c + dc]); break
                        else:
                            return None
            return sh, tabd

        res = None
        for k in (8, 16, 32, 64):
            res = lattice(lambda dr, dc, k=k: max(12 * 8 // k, (h - abs(dr)) * (w - abs(dc)) // k))
            if res is not None:
                break
        if res is None:
            res = lattice(lambda dr, dc: 1)
        shifts, tabd = res
        truth = lambda r, c: tabd[(r, c)]
    shifts.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s))

    def intact_source(r0, c0, dh, dw):
        for dr, dc in shifts:
            sr, sc = r0 + dr, c0 + dc
            if 0 <= sr and sr + dh < h and 0 <= sc and sc + dw < w:
                if clean[sr:sr + dh + 1, sc:sc + dw + 1].all():
                    return sr, sc
        return None

    # --- noise patches = connected blobs of the noise colour in I
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

    def repair_block(r0, c0, dh, dw, src):
        sr, sc = src
        blk = [(r0 + i, c0 + j) for i in range(dh + 1) for j in range(dw + 1)]
        # Paste is transparent for 0: noise cells whose tile colour is 0 are blanked first
        zc = [(r, c) for r, c in blk if remaining[r, c] and truth(r, c) == 0 and cur[r, c] != 0]
        if zc:
            ops.append(0); sels.append(sel_of(zc))
            for r, c in zc:
                cur[r, c] = 0
        if any(remaining[r, c] and truth(r, c) != 0 and cur[r, c] != truth(r, c) for r, c in blk):
            ops.append(28); sels.append([sr, sc, dh, dw])     # CopyI: intact block whole tiles away (full rectangle)
            ops.append(30); sels.append([r0, c0, 0, 0])       # Paste it over the patch
            for i in range(dh + 1):
                for j in range(dw + 1):
                    v = I[sr + i, sc + j]
                    if v != 0:
                        cur[r0 + i, c0 + j] = v
        for r, c in blk:
            remaining[r, c] = False

    for comp in comps:                       # one noise patch at a time
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        r0, c0 = min(rs), min(cs)
        dh, dw = max(rs) - r0, max(cs) - c0
        if not any(remaining[r, c] for r, c in comp):
            continue
        src = intact_source(r0, c0, dh, dw)
        if src is not None:                  # the whole patch in one copy
            repair_block(r0, c0, dh, dw, src)
            continue
        # patch too large for a single intact copy: largest intact sub-blocks
        while True:
            todo = [p for p in comp if remaining[p[0], p[1]]]
            if not todo:
                break
            sr0, sc0 = todo[0]
            best = None
            for hh in range(1, max(rs) - sr0 + 2):
                for ww in range(1, max(cs) - sc0 + 2):
                    cov = int(remaining[sr0:sr0 + hh, sc0:sc0 + ww].sum())
                    if best is not None and (cov, -hh * ww) <= best[0]:
                        continue
                    s = intact_source(sr0, sc0, hh - 1, ww - 1)
                    if s is not None:
                        best = ((cov, -hh * ww), hh - 1, ww - 1, s)
            if best is None:
                break
            _, bh, bw, s = best
            repair_block(sr0, sc0, bh, bw, s)

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
