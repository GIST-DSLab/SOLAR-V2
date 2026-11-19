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
from random import choice, randint, sample
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of
from dsl import *
from utils import unifint


def sample_colors(num_examples=None) -> dict:
    # The only colour role the rule reads is the noise colour; the wallpaper
    # colours may vary freely per instance.
    return {"noisec": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if noisec is None:
        noisec = choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
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
    if any(v == -1 for row in go for v in row):
        raise ValueError('wallpaper does not cover the canvas')
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Rule: the grid is a wallpaper on a translation lattice; patches of one
    # noise colour hide parts of it.  Unit of work: ONE WALLPAPER COLOUR AT A
    # TIME over the whole grid -- for each colour c, a single Color(c) op paints
    # every hidden cell whose lattice-equivalent visible cell in I is c.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # noise colour: present in every demo input, absent from every demo output
    noisec = None
    if examples:
        cand = set(range(10))
        for ei, eo in examples:
            cand &= set(np.unique(np.asarray(ei)).tolist()) - set(np.unique(np.asarray(eo)).tolist())
        cand &= set(np.unique(I).tolist())
        if len(cand) == 1:
            noisec = cand.pop()
    if noisec is None:
        # fallback (verifier heuristic): fewest components, then fewest cells
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
    if not noise.any():
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    clean = ~noise

    # translation lattice of the wallpaper, measured on visible cells of I
    lattice = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if dr == 0 and dc == 0:
                continue
            r0, r1 = max(0, -dr), min(h, h - dr)
            c0, c1 = max(0, -dc), min(w, w - dc)
            if r1 <= r0 or c1 <= c0:
                continue
            A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            m = clean[r0:r1, c0:c1] & clean[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            if int(m.sum()) < max(12, (r1 - r0) * (c1 - c0) // 8):
                continue
            if (A[m] == B[m]).all():
                lattice.append((abs(dr) + abs(dc), dr, dc))
    lattice.sort()

    # wallpaper colour under each hidden cell, read from a lattice-equivalent
    # visible cell of I (propagating through already-resolved cells if needed)
    known = clean.copy()
    val = I.copy()
    todo = [(r, c) for r in range(h) for c in range(w) if noise[r, c]]
    progress = True
    while todo and progress:
        progress = False
        nxt = []
        for (r, c) in todo:
            got = None
            for _, dr, dc in lattice:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and known[rr, cc]:
                    got = val[rr, cc]; break
            if got is None:
                nxt.append((r, c))
            else:
                val[r, c] = got; known[r, c] = True; progress = True
        todo = nxt

    # one Color op per wallpaper colour: all hidden cells of that colour at once
    groups = {}
    for r in range(h):
        for c in range(w):
            if noise[r, c] and known[r, c] and val[r, c] != noisec:
                groups.setdefault(int(val[r, c]), []).append((r, c))
    for col in sorted(groups, key=lambda k: -len(groups[k])):
        ops.append(col); sels.append(sel_of(groups[col]))

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
