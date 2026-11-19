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


def sample_colors(num_examples=None) -> dict:
    bgc, noisec = random.sample(list(range(10)), 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, noisec=None, **kw) -> dict:
    from dsl import (interval, canvas, asindices, remove, paint, shift, fill,
                     backdrop, apply, lbind, rbind, toobject, normalize, sfilter,
                     dmirror, identity, rot90, rot180, rot270)
    from utils import unifint
    cols = interval(0, 10, 1)
    if bgc is None or noisec is None:
        bgc, noisec = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, numc)
    pobj = frozenset({(random.choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h // hp + 1):
        for b in range(w // wp + 1):
            loci = (a + 1) + hp * a
            locj = (b + 1) + wp * b
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
        ph = random.randint(2, 6)
        pw = random.randint(2, 6)
        loci = random.randint(0, h - ph)
        locj = random.randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = random.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


# ---------------------------------------------------------------- helpers
def _period(G, valid, axis):
    """Smallest period along `axis` under which every undamaged cell agrees
    with every other undamaged cell of its residue class."""
    if axis == 1:
        G, valid = G.T, valid.T
    n = G.shape[0]
    for p in range(1, n + 1):
        ok = True
        for k in range(p):
            sub, vs = G[k::p], valid[k::p]
            for j in range(G.shape[1]):
                if len(set(sub[vs[:, j], j].tolist())) > 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return n


def _measure(I):
    """Damage colour = the colour whose removal leaves the smallest fully
    determined wallpaper tile.  Returns (nc, vp, hp, T) or None."""
    h, w = I.shape
    best = None
    for c in sorted(set(I.flatten().tolist())):
        valid = I != c
        if valid.all():
            continue
        vp = _period(I, valid, 0)
        hp = _period(I, valid, 1)
        if vp >= h or hp >= w:
            continue
        T = np.full((vp, hp), -1, dtype=int)
        for r in range(h):
            for cc in range(w):
                if valid[r, cc]:
                    T[r % vp, cc % hp] = I[r, cc]
        if (T < 0).any():
            continue
        key = (vp * hp, int((~valid).sum()))
        if best is None or key < best[0]:
            best = (key, c, vp, hp, T)
    if best is None:
        return None
    return best[1:]


def _lattice(I):
    """Return (nc, vp, hp, T, known, lattice_options).  T is the vp x hp
    period tile read from the undamaged cells; lattice options are the
    (row residue, col residue, colour) of full single-colour lines."""
    nc, vp, hp, T = _measure(I)
    opts = []
    for a in range(vp):
        row = T[a, :]
        if (row < 0).any() or len(set(row.tolist())) != 1:
            continue
        for b in range(hp):
            col = T[:, b]
            if (col < 0).any() or len(set(col.tolist())) != 1:
                continue
            if int(col[0]) == int(row[0]):
                opts.append((a, b, int(row[0])))
    return nc, vp, hp, T, opts


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    if _measure(I) is None:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    nc, vp, hp, T, opts = _lattice(I)
    G = I.copy()
    noise = (I == nc)

    def emit(op, cells):
        cells = [(r, c) for r, c in cells if G[r, c] != op]
        if not cells:
            return
        ops.append(op); sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = op

    if noise.any() and not (T < 0).all():
        # choose the lattice (separator row/col residues) for which some full
        # tile cell (the core between lines) is undamaged in I
        chosen = None
        for a, b, lc in opts:
            ch, cw = vp - 1, hp - 1
            for sr in range((a + 1) % vp, h - ch + 1, vp):
                for sc in range((b + 1) % hp, w - cw + 1, hp):
                    if not noise[sr:sr + ch, sc:sc + cw].any():
                        chosen = (a, b, lc, sr, sc)
                        break
                if chosen:
                    break
            if chosen:
                break

        if chosen is not None:
            a, b, lc, csr, csc = chosen
            ch, cw = vp - 1, hp - 1
            # 1. redraw the grid lines: damaged cells on each separator row,
            #    then each separator column, are repainted the line colour
            for r in range(a, h, vp):
                emit(lc, [(r, c) for c in range(w) if G[r, c] == nc])
            for c in range(b, w, hp):
                emit(lc, [(r, c) for r in range(h) if G[r, c] == nc])
            # 2. stamp the intact tile cell into every damaged lattice cell
            jobs = {}
            for tr0 in range((a + 1) - vp * ((a + 1) // vp + 1), h, vp):
                for tc0 in range((b + 1) - hp * ((b + 1) // hp + 1), w, hp):
                    r0, r1 = max(tr0, 0), min(tr0 + ch, h)
                    c0, c1 = max(tc0, 0), min(tc0 + cw, w)
                    if r0 >= r1 or c0 >= c1:
                        continue
                    if not (G[r0:r1, c0:c1] == nc).any():
                        continue
                    key = (r0 - tr0, c0 - tc0, r1 - r0, c1 - c0)
                    jobs.setdefault(key, []).append((r0, c0))
            # full-size cells first, then each clipped edge variant
            order = sorted(jobs, key=lambda k: (-(k[2] * k[3]), k))
            for (dr, dc, hh, ww) in order:
                sr, sc = csr + dr, csc + dc
                src = I[sr:sr + hh, sc:sc + ww]
                ops.append(28); sels.append([sr, sc, hh - 1, ww - 1])  # full rect of intact cell
                for (r0, c0) in jobs[(dr, dc, hh, ww)]:
                    before = G.copy()
                    reg = G[r0:r0 + hh, c0:c0 + ww]
                    nz = src != 0
                    reg[nz] = src[nz]
                    if not np.array_equal(before, G):
                        ops.append(30); sels.append([r0, c0, 0, 0])
                    # 0-valued tile cells cannot be pasted: paint the
                    # damage left inside the landing area with colour 0
                    emit(0, [(r, c) for r in range(r0, r0 + hh)
                             for c in range(c0, c0 + ww) if G[r, c] == nc])
                # drop a CopyI whose pastes all turned out to be no-ops
                if ops[-1] == 28:
                    ops.pop(); sels.pop()

        # safety net: residual damage painted from the period tile, by colour
        rest = {}
        for r in range(h):
            for c in range(w):
                if G[r, c] == nc and T[r % vp, c % hp] >= 0:
                    rest.setdefault(int(T[r % vp, c % hp]), []).append((r, c))
        for col in sorted(rest):
            emit(col, rest[col])

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
