"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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
    bgc, noisec, objc = random.sample(range(10), 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, objc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return a + int(round(d * (b - a)))

    if bgc is None or noisec is None or objc is None:
        bgc, noisec, objc = random.sample(range(10), 3)
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    nbh = [(0, 0), (1, 0), (0, 1), (1, 1)]
    offs = [(0, 0), (-1, 0), (0, -1), (-1, -1)]

    def ok(obj):
        for (i, j) in obj:
            for (a, b) in offs:
                if sum((i + a + x, j + b + y) in obj for x, y in nbh) < 4:
                    return True
        return False

    def irreducible(obj):
        # the shape must not itself be an upscaled shape, else its scale is ambiguous
        hh = max(i for i, _ in obj) + 1
        ww = max(j for _, j in obj) + 1
        m = np.zeros((hh, ww), dtype=bool)
        for i, j in obj:
            m[i, j] = True
        for k in range(2, min(hh, ww) + 1):
            if hh % k or ww % k:
                continue
            sm = m.reshape(hh // k, k, ww // k, k).any(axis=(1, 3))
            if np.array_equal(sm.repeat(k, 0).repeat(k, 1), m):
                return False
        return True

    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = [(i, j) for i in range(oh) for j in range(ow)]
    while True:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = random.choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        obj = set(random.sample(bounds, ncells))
        if not ok(obj):
            continue
        mi = min(i for i, _ in obj)
        mj = min(j for _, j in obj)
        obj = {(i - mi, j - mj) for i, j in obj}
        if irreducible(obj):
            break
    oh = max(i for i, _ in obj) + 1
    ow = max(j for _, j in obj) + 1
    go = np.full((oh, ow), bgc, dtype=int)
    for (i, j) in obj:
        go[i, j] = noisec
    fmax = min((max_h - 2) // oh, (max_w - 2) // ow, 28 // oh, 28 // ow)
    if fmax < 2:
        raise ValueError("grid too small")
    fac = unifint(diff_lb, diff_ub, (2, fmax))
    big = np.where(go == noisec, objc, bgc).repeat(fac, 0).repeat(fac, 1)
    bh, bw = big.shape
    h = unifint(diff_lb, diff_ub, (bh + 2, max_h))
    w = unifint(diff_lb, diff_ub, (bw + 2, max_w))
    loci = random.randint(1, h - bh - 1)
    locj = random.randint(1, w - bw - 1)
    gi = np.full((h, w), bgc, dtype=int)
    for i in range(bh):
        for j in range(bw):
            if big[i, j] == objc:
                gi[loci + i, locj + j] = objc
    cands = [(i, j) for i in range(h) for j in range(w) if gi[i, j] == bgc]
    namt = unifint(diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
    for (i, j) in random.sample(cands, namt):
        gi[i, j] = noisec
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: dominant colour of the border ring (the object never touches it)
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    # object colour: made of solid blocks (no isolated cell); noise has isolated cells
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(1 for (r, cc) in cs
                  if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))))
        return (1 if iso == 0 else 0, len(cs))

    objc = max(others, key=block_score)
    noisec = [c for c in others if c != objc][0]

    # upscaled object bbox and its scale factor (largest consistent block size)
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    H, W = r1 - r0 + 1, c1 - c0 + 1
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = 1
    for k in range(min(H, W), 0, -1):
        if H % k or W % k:
            continue
        small = M.reshape(H // k, k, W // k, k).any(axis=(1, 3))
        if np.array_equal(small.repeat(k, 0).repeat(k, 1), M):
            fac = k
            break
    ph, pw = H // fac, W // fac
    P = M[::fac, ::fac]

    ops, sels = [], []
    G = I.copy()                                   # working grid as it stands
    clip = {"g": None}                             # clipboard contents (CopyO)
    lat_r = [r0 + i * fac for i in range(ph)]      # one representative row per block
    lat_c = [c0 + j * fac for j in range(pw)]      # one representative column per block

    # 1. stray noise sitting on the sampling lattice of the object box -> background
    nz = [(r, c) for r in lat_r for c in lat_c if G[r, c] == noisec]
    if nz:
        ops.append(bgc); sels.append(sel_of(nz))
        for r, c in nz:
            G[r, c] = bgc
    # 2. the (upscaled) object takes the noise colour
    obj = [tuple(x) for x in np.argwhere(I == objc).tolist()]
    ops.append(noisec); sels.append(sel_of(obj))
    for r, c in obj:
        G[r, c] = noisec

    def paste_sim(clipg, r, c):
        out = G.copy()
        ch, cw = clipg.shape
        for a in range(ch):
            for b in range(cw):
                if clipg[a, b] != 0 and r + a < hi and c + b < wi:
                    out[r + a, c + b] = clipg[a, b]
        return out

    def compact(src_box, dst_rc, relevant):
        # copy one representative line (src_box = r, c, h, w) onto dst_rc; only the
        # `relevant` destination cells are read again later
        sr, sc, sh, sw = src_box
        dr, dc = dst_rc
        want = {(dr + a, dc + b): int(G[sr + a, sc + b])
                for a in range(sh) for b in range(sw) if (dr + a, dc + b) in relevant}
        if all(G[p] == v for p, v in want.items()):
            return
        # a transparent paste cannot write 0: if a 0 must land on a non-0 cell,
        # clear the destination line's read cells first
        if any(v == 0 and G[p] != 0 for p, v in want.items()):
            line = sorted(want)
            ops.append(0); sels.append(sel_of(line))
            for p in line:
                G[p] = 0
        if all(G[p] == v for p, v in want.items()):
            return
        reuse = clip["g"] is not None and clip["g"].shape == (sh, sw)
        if reuse:
            after = paste_sim(clip["g"], dr, dc)
            reuse = all(after[p] == v for p, v in want.items())
        if not reuse:
            ops.append(29); sels.append([sr, sc, sh - 1, sw - 1])
            clip["g"] = G[sr:sr + sh, sc:sc + sw].copy()
            after = paste_sim(clip["g"], dr, dc)
        ops.append(30); sels.append([dr, dc, 0, 0])
        G[:, :] = after

    # 3. downscale in place: representative rows up under the first block row ...
    for i in range(1, ph):
        compact((lat_r[i], c0, 1, W), (r0 + i, c0), {(r0 + i, c) for c in lat_c})
    # ... then representative columns left beside the first block column
    rows = range(r0, r0 + ph)
    for j in range(1, pw):
        compact((r0, lat_c[j], ph, 1), (r0, c0 + j), {(r, c0 + j) for r in rows})

    # 4. keep only the downscaled object
    ops.append(33); sels.append([r0, c0, ph - 1, pw - 1])

    # the answer as read from I alone: downscaled object in the noise colour
    target = np.where(P, noisec, bgc)

    def run(seq):
        g = I.copy()
        cl = None
        for op, sel in seq:
            if isinstance(sel, dict):
                cells_ = [tuple(x) for x in sel["cells"]]
            else:
                r, c, h, w = sel
                cells_ = [(r + a, c + b) for a in range(h + 1) for b in range(w + 1)]
            if op <= 9:
                for p in cells_:
                    g[p] = op
            elif op == 29:
                r, c, h, w = sel
                cl = g[r:r + h + 1, c:c + w + 1].copy()
            elif op == 30:
                if cl is not None:
                    r, c = sel[0], sel[1]
                    for a in range(cl.shape[0]):
                        for b in range(cl.shape[1]):
                            if cl[a, b] != 0 and r + a < g.shape[0] and c + b < g.shape[1]:
                                g[r + a, c + b] = cl[a, b]
            elif op == 33:
                r, c, h, w = sel
                g = g[r:r + h + 1, c:c + w + 1].copy()
        return g

    # drop any op whose removal still yields the I-derived answer
    seq = list(zip(ops, sels))
    changed = True
    while changed:
        changed = False
        for k in range(len(seq)):
            red = seq[:k] + seq[k + 1:]
            out = run(red)
            if out.shape == target.shape and np.array_equal(out, target):
                seq = red
                changed = True
                break
    ops = [o for o, _ in seq] + [34]
    sels = [s_ for _, s_ in seq] + [[0, 0, ph - 1, pw - 1]]
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
