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

try:
    from maker.sel_helpers import sel_of
except Exception:
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for (r, c) in cells]}


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    noisec = choice(cols)                      # colour the damage patches are painted with
    remcols = [c for c in cols if c != noisec]
    numc = randint(2, 9)                       # how many colours the repeating tile uses
    ccols = sample(tuple(remcols), numc)
    return {"noisec": noisec, "ccols": tuple(ccols)}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             noisec=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if noisec is None:
        noisec = choice(cols)
    if ccols is None:
        remcols = remove(noisec, cols)
        numc = unifint(diff_lb, diff_ub, (2, 9))
        ccols = sample(remcols, numc)
    ccols = tuple(ccols)

    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    hp = unifint(diff_lb, diff_ub, (2, max(2, h // 2 - 1)))
    wp = unifint(diff_lb, diff_ub, (2, max(2, w // 2 - 1)))
    pinds = asindices(canvas(-1, (hp, wp)))
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
    numpatches = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
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
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and \
               len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """The grid is one periodic wallpaper with rectangular blocks knocked out and
    repainted in a single 'damage' colour.  Repair = copy an undamaged block that
    sits one period away and paste it over the hole; only what a transparent paste
    cannot carry (colour 0) is painted afterwards."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    full = [0, 0, H - 1, W - 1]

    diff_cells = [(r, c) for r in range(H) for c in range(W) if I[r, c] != O[r, c]]
    if not diff_cells:                       # nothing was damaged
        ops.append(34); sels.append(full)
        return ops, sels

    noisec = Counter(int(I[r, c]) for r, c in diff_cells).most_common(1)[0][0]
    noise = (I == noisec)

    # --- translation vectors under which the undamaged content of I is invariant ---
    def cons_v(p):
        for r in range(H - p):
            for c in range(W):
                if not noise[r, c] and not noise[r + p, c] and I[r, c] != I[r + p, c]:
                    return False
        return True

    def cons_h(p):
        for c in range(W - p):
            for r in range(H):
                if not noise[r, c] and not noise[r, c + p] and I[r, c] != I[r, c + p]:
                    return False
        return True

    vs = [p for p in range(1, H) if cons_v(p)]
    hs = [p for p in range(1, W) if cons_h(p)]
    drs = [0] + vs + [-p for p in vs]
    dcs = [0] + hs + [-p for p in hs]
    cands = sorted([(dr, dc) for dr in drs for dc in dcs if not (dr == 0 and dc == 0)],
                   key=lambda t: (abs(t[0]) + abs(t[1]), abs(t[0])))

    cur = I.copy()
    to_paint = []

    def find_src(r0, c0, h, w):
        """an undamaged block of the same shape, one lattice period away"""
        for dr, dc in cands:
            sr, sc = r0 + dr, c0 + dc
            if sr < 0 or sc < 0 or sr + h > H or sc + w > W:
                continue
            src = I[sr:sr + h, sc:sc + w]
            if np.any(src == noisec):
                continue
            if np.array_equal(src, O[r0:r0 + h, c0:c0 + w]):
                return sr, sc
        return None

    def handle(r0, c0, h, w):
        sub = cur[r0:r0 + h, c0:c0 + w]
        if not np.any(sub == noisec):
            return
        if h * w >= 2:
            src = find_src(r0, c0, h, w)
            if src is not None:
                sr, sc = src
                patch = I[sr:sr + h, sc:sc + w]
                after = np.where(patch != 0, patch, sub)
                if not np.array_equal(after, sub):
                    ops.append(28); sels.append([sr, sc, h - 1, w - 1])   # CopyI: intact copy of this block
                    ops.append(30); sels.append([r0, c0, 0, 0])           # Paste it over the damaged block
                    cur[r0:r0 + h, c0:c0 + w] = after
                for r in range(r0, r0 + h):                               # what the paste could not carry
                    for c in range(c0, c0 + w):
                        if cur[r, c] == noisec:
                            to_paint.append((r, c))
                return
        if h > 1:                                                          # no intact twin this size: halve
            half = h // 2
            handle(r0, c0, half, w); handle(r0 + half, c0, h - half, w)
        elif w > 1:
            half = w // 2
            handle(r0, c0, h, half); handle(r0, c0 + half, h, w - half)
        else:
            if cur[r0, c0] == noisec:
                to_paint.append((r0, c0))

    # --- damaged blocks = connected regions of the damage colour ---
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for r in range(H):
        for c in range(W):
            if noise[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    a, b = q.popleft(); cells.append((a, b))
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        na, nb = a + da, b + db
                        if 0 <= na < H and 0 <= nb < W and noise[na, nb] and not seen[na, nb]:
                            seen[na, nb] = True; q.append((na, nb))
                comps.append(cells)
    comps.sort(key=lambda cs: (-len(cs), min(cs)))

    for cells in comps:
        del to_paint[:]
        rs = [a for a, _ in cells]; cs = [b for _, b in cells]
        r0, c0 = min(rs), min(cs)
        h, w = max(rs) - r0 + 1, max(cs) - c0 + 1
        handle(r0, c0, h, w)
        by_col = {}
        for (r, c) in to_paint:                       # cells still showing the damage colour
            if cur[r, c] != O[r, c]:
                by_col.setdefault(int(O[r, c]), []).append((r, c))
        for col in sorted(by_col):
            pts = by_col[col]
            ops.append(col); sels.append(sel_of(pts))
            for (r, c) in pts:
                cur[r, c] = col

    ops.append(34); sels.append(full)
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
