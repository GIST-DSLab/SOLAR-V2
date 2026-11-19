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
from math import gcd
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # the noise colour is the rule's key convention -> fixed per episode
    noisec = random.choice(range(10))
    return {"noisec": noisec}


def _generate_once(diff_lb, diff_ub, max_h, max_w, noisec=None) -> dict:
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
    # extended ranges so the lattice tiling covers the whole canvas (no -1 left)
    for a in range(-2, 2 * (h // hp + 1) + (ofs * (w // wp + 2)) // hp + 2):
        for b in range(-1, w // wp + 2):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    if any(v == -1 for row in go for v in row):
        raise ValueError("incomplete tiling")
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
    return {'input': gi, 'output': go}


def _recoverable(gi, go, noisec):
    """Every noise cell must have a clean periodic copy somewhere in the grid."""
    I = np.asarray(gi); O = np.asarray(go)
    h, w = O.shape
    known = I != noisec
    tv = []
    for dr in range(-h + 1, h):
        for dc in range(-w + 1, w):
            if (dr, dc) == (0, 0):
                continue
            r0, r1, c0, c1 = max(0, -dr), min(h, h - dr), max(0, -dc), min(w, w - dc)
            if (O[r0:r1, c0:c1] == O[r0 + dr:r1 + dr, c0 + dc:c1 + dc]).all():
                tv.append((dr, dc))
    for r, c in zip(*np.nonzero(~known)):
        if not any(0 <= r + a < h and 0 <= c + b < w and known[r + a, c + b] for a, b in tv):
            return False
    return True


def generate(diff_lb, diff_ub, max_h, max_w, noisec=None) -> dict:
    if noisec is None:
        noisec = random.choice(range(10))
    for _ in range(50):
        try:
            g = _generate_once(diff_lb, diff_ub, max_h, max_w, noisec=noisec)
        except ValueError:
            continue
        if _recoverable(g['input'], g['output'], noisec):
            return g
    raise ValueError("could not build a recoverable instance")


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # --- noise colour: the convention shown by the demonstrations (in inputs, never in outputs)
    noisec = None
    if examples:
        cand = set(range(10))
        seen = set()
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            cand -= set(np.unique(eo).tolist())
            seen |= set(np.unique(ei).tolist())
        cand &= seen
        if len(cand) == 1:
            noisec = cand.pop()
        elif cand:
            present = [c for c in cand if (I == c).any()]
            if present:
                noisec = present[0]
    if noisec is None:
        # fallback: colour with fewest objects, then fewest cells (verifier rule)
        pal = sorted(set(I.flatten().tolist()))
        objs = objects(tuple(tuple(int(v) for v in r) for r in I), True, False, False)
        nobj = {c: sum(1 for o in objs if color(o) == c) for c in pal}
        mn = min(nobj.values())
        noisec = min([c for c in pal if nobj[c] == mn], key=lambda c: (I == c).sum())

    noise = (I == noisec)
    ops, sels = [], []
    if noise.any():
        known = ~noise
        # --- candidate period translations: shifts under which no pair of clean cells disagrees
        cands = []
        for dr in range(0, h):
            for dc in range(-(w - 1), w):
                if dr == 0 and dc <= 0:
                    continue
                r0, r1 = max(0, -dr), min(h, h - dr)
                c0, c1 = max(0, -dc), min(w, w - dc)
                A = I[r0:r1, c0:c1]; B = I[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                m = known[r0:r1, c0:c1] & known[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
                if int(m.sum()) < 3 or (A[m] != B[m]).any():
                    continue
                cands.append((dr * dr + dc * dc, dr, dc))
        cands.sort()
        allc = [(dr, dc) for _, dr, dc in cands]
        cands = allc[:60]
        kr, kc = np.nonzero(known)
        kv = I[kr, kc]
        nr, nc = np.nonzero(noise)

        def hnf(v1, v2):
            x, y = list(v1), list(v2)
            if x[0] < 0: x = [-x[0], -x[1]]
            if y[0] < 0: y = [-y[0], -y[1]]
            while y[0] != 0:
                q = x[0] // y[0]
                x, y = y, [x[0] - q * y[0], x[1] - q * y[1]]
            return x[0], x[1], abs(y[1])

        def classify(a, b, d):
            def rep(rr, cc):
                k = rr // a
                return (rr - k * a) * d + (cc - k * b) % d
            return rep

        BIG = 4 * (h + w) * (h + w) + 64

        def rank1(v):
            a, b = v
            if a == 0:
                return lambda rr, cc: rr * BIG + cc % b
            return lambda rr, cc: (rr % a) * BIG + (cc - (rr // a) * b) + 2 * BIG // 4

        def evaluate(rep):
            kid = rep(kr, kc)
            nid = rep(nr, nc)
            uk, inv = np.unique(kid, return_inverse=True)
            lut = np.full(len(uk), -1)
            lut[inv] = kv
            if (lut[inv] != kv).any():
                return None          # lattice identifies two clean cells of different colour
            pos = np.clip(np.searchsorted(uk, nid), 0, len(uk) - 1)
            hit = uk[pos] == nid
            return np.where(hit, lut[pos], -1)

        # --- choose the lattice consistent with every clean cell that restores the most
        #     noise cells (ties: the finest lattice)
        state = {"best": None}

        def consider(det, rep):
            got = evaluate(rep)
            if got is None:
                return None
            key = (-int((got >= 0).sum()), det)
            if state["best"] is None or key < state["best"][0]:
                state["best"] = (key, got)
            return key

        def pair(v1, v2):
            det = abs(v1[0] * v2[1] - v1[1] * v2[0])
            if det == 0:
                return
            a, b, d = hnf(v1, v2)
            if a > 0 and d > 0:
                consider(det, classify(a, b, d))

        r1 = []
        for i, v in enumerate(allc):
            k = consider((10 ** 9) + i, rank1(v))
            if k is not None:
                r1.append((k, v))
        r1.sort()
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                pair(cands[i], cands[j])
        for _, v1 in r1[:4]:
            for v2 in allc:
                pair(v1, v2)
        best = state["best"]
        val = I.copy()
        if best is not None:
            got = best[1]
            for r, c, g in zip(nr, nc, got):
                if g >= 0:
                    val[r, c] = g
        # --- emit per noise patch (connected component), one Color per restored colour
        seen = np.zeros_like(noise)
        for r, c in zip(*np.nonzero(noise)):
            if seen[r, c]:
                continue
            comp = []
            stack = [(r, c)]; seen[r, c] = True
            while stack:
                a, b = stack.pop(); comp.append((a, b))
                for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                    if 0 <= na < h and 0 <= nb < w and noise[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True; stack.append((na, nb))
            bycol = {}
            for a, b in comp:
                bycol.setdefault(int(val[a, b]), []).append((a, b))
            for col in sorted(bycol, key=lambda k: -len(bycol[k])):
                if col == noisec:
                    continue
                ops.append(col); sels.append(sel_of(bycol[col]))
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
