"""
ARC Task: 29ec7d0e (RE-ARC) — LLM-generated grid_maker
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


def _unif(diff_lb, diff_ub, lo, hi):
    if hi <= lo:
        return lo
    a = lo + (hi - lo) * diff_lb
    b = lo + (hi - lo) * diff_ub
    a, b = int(round(a)), int(round(b))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if b < a:
        a, b = b, a
    return random.randint(a, b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec = random.sample(cols, 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, **kw) -> dict:
    cols = list(range(10))
    h = _unif(diff_lb, diff_ub, 10, max(10, max_h))
    w = _unif(diff_lb, diff_ub, 10, max(10, max_w))
    hp = _unif(diff_lb, diff_ub, 2, h // 2 - 1)
    wp = _unif(diff_lb, diff_ub, 2, w // 2 - 1)
    remcols = [c for c in cols if c != noisec]
    numc = _unif(diff_lb, diff_ub, 2, 9)
    ccols = random.sample(remcols, numc)
    pat = np.array([[random.choice(ccols) for _ in range(wp)] for _ in range(hp)])
    go = np.full((h, w), bgc, dtype=int)
    locs = []
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            li = (a + 1) + hp * a
            lj = (b + 1) + wp * b
            locs.append((li, lj))
            r1, c1 = min(h, li + hp), min(w, lj + wp)
            if li < h and lj < w:
                go[li:r1, lj:c1] = pat[:r1 - li, :c1 - lj]
    full_locs = [(li, lj) for li, lj in locs if li + hp <= h and lj + wp <= w]
    numpatches = _unif(diff_lb, diff_ub, 1, (h * w) // 20)
    gi = go.copy()
    succ, tr = 0, 0
    while succ < numpatches and tr < 5 * numpatches:
        tr += 1
        ph, pw = random.randint(2, 6), random.randint(2, 6)
        li, lj = random.randint(0, h - ph), random.randint(0, w - pw)
        gi2 = gi.copy()
        gi2[li:li + ph, lj:lj + pw] = noisec
        nz = gi2 == noisec
        if not any(not nz[a:a + hp, b:b + wp].any() for a, b in full_locs):
            continue
        if (~nz.any(axis=1)).sum() >= 2 and (~nz.any(axis=0)).sum() >= 2:
            succ += 1
            gi = gi2
    if succ == 0:
        raise ValueError("no noise placed")
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _period(G, ok, axis):
    # smallest shift p along axis such that every pair of non-noise cells p apart agrees
    if axis == 1:
        G, ok = G.T, ok.T
    n = G.shape[0]
    for p in range(1, n):
        a, b = G[:-p], G[p:]
        m = ok[:-p] & ok[p:]
        if np.all(a[m] == b[m]):
            return p
    return n


def _noise_colour(I, examples):
    if examples:
        cand = None
        for ei, eo in examples:
            ei, eo = np.asarray(ei), np.asarray(eo)
            s = set(np.unique(ei).tolist()) - set(np.unique(eo).tolist())
            cand = s if cand is None else cand & s
        if cand and len(cand) == 1:
            return cand.pop()
    # verifier fallback: fewest connected objects, then fewest cells
    from collections import Counter
    h, w = I.shape
    seen = np.zeros_like(I, dtype=bool)
    nobj = Counter()
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            v = I[r, c]; nobj[v] += 1
            st = [(r, c)]; seen[r, c] = True
            while st:
                y, x = st.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and not seen[yy, xx] and I[yy, xx] == v:
                        seen[yy, xx] = True; st.append((yy, xx))
    cnt = Counter(I.flatten().tolist())
    m = min(nobj.values())
    return min([v for v in nobj if nobj[v] == m], key=lambda v: cnt[v])


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    nc = int(_noise_colour(I, examples))
    noise = I == nc
    ok = ~noise
    vp = _period(I, ok, 0)
    hp = _period(I, ok, 1)

    # one clean period tile, read from the intact (non-noise) cells of I
    T = np.full((vp, hp), -1, dtype=int)
    for r in range(h):
        for c in range(w):
            if ok[r, c] and T[r % vp, c % hp] < 0:
                T[r % vp, c % hp] = I[r, c]

    # lattice phase: each tile starts at a uniform (separator) row / column of the tile
    pr = next((i for i in range(vp) if len(set(T[i].tolist())) == 1), 0)
    pc = next((j for j in range(hp) if len(set(T[:, j].tolist())) == 1), 0)
    rb = sorted({0, h} | {x for x in range(pr % vp, h, vp)})
    cb = sorted({0, w} | {x for x in range(pc % hp, w, hp)})

    G = I.copy()
    last_copy = None
    # unit of work: one lattice tile at a time
    for a in range(len(rb) - 1):
        for b in range(len(cb) - 1):
            r0, r1, c0, c1 = rb[a], rb[a + 1], cb[b], cb[b + 1]
            if not noise[r0:r1, c0:c1].any():
                continue                      # this tile is intact
            bh, bw = r1 - r0, c1 - c0
            # nearest lattice tile (same phase, same shape) that is fully intact in I
            cand = []
            for i in range(-(h // vp) - 1, h // vp + 2):
                for j in range(-(w // hp) - 1, w // hp + 2):
                    if i == 0 and j == 0:
                        continue
                    sr, sc = r0 + i * vp, c0 + j * hp
                    if sr < 0 or sc < 0 or sr + bh > h or sc + bw > w:
                        continue
                    if noise[sr:sr + bh, sc:sc + bw].any():
                        continue
                    cand.append((abs(i) + abs(j), sr, sc))
            if cand:
                _, sr, sc = min(cand)
                src = I[sr:sr + bh, sc:sc + bw]
                newG = G.copy()
                blk = newG[r0:r1, c0:c1]
                blk[src != 0] = src[src != 0]
                if not np.array_equal(newG, G):
                    if last_copy != (sr, sc, bh, bw):
                        # full rectangle: a whole intact tile one or more periods away
                        ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])
                        last_copy = (sr, sc, bh, bw)
                    ops.append(30); sels.append([r0, c0, 0, 0])
                    G = newG
            else:
                # no intact twin of this tile: repaint its noise cells colour by colour
                cells = {}
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        if noise[r, c]:
                            v = int(T[r % vp, c % hp])
                            if v != 0 and G[r, c] != v:
                                cells.setdefault(v, []).append((r, c))
                for v in sorted(cells):
                    ops.append(v); sels.append(sel_of(cells[v]))
                    for (r, c) in cells[v]:
                        G[r, c] = v

    # any cells still in the noise colour sit on tile cells whose value is 0
    # (Paste can't write 0), so one Color0 over the remaining noise colour fixes them
    if nc != 0:
        left = [(r, c) for r in range(h) for c in range(w) if G[r, c] == nc]
        if left:
            ops.append(0); sels.append(sel_of(left))
            for r, c in left:
                G[r, c] = 0

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
                        f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29ec7d0e"
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
                                f"for task 29ec7d0e"
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
                    f"Failed to build a complete episode for task 29ec7d0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29ec7d0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
