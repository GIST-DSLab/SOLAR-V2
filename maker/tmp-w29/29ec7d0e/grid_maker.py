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

import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of
try:
    from utils import unifint, interval, canvas, asindices, remove, choice, sample, randint
except Exception:
    pass


def sample_colors(num_examples=None) -> dict:
    bgc, noisec = random.sample(range(10), 2)
    return {"bgc": bgc, "noisec": noisec}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    hp = unifint(diff_lb, diff_ub, (2, h // 2 - 1))
    wp = unifint(diff_lb, diff_ub, (2, w // 2 - 1))
    pinds = asindices(canvas(-1, (hp, wp)))
    remcols = [c for c in range(10) if c != noisec]
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
    succ, tr, maxtr = 0, 0, 5 * numpatches
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
    rotf = random.choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {"input": gi, "output": go}


def _period(I, known, axis):
    """Smallest period p along `axis` such that every residue line holds one
    consistent value over all non-noise cells, and every residue is observed."""
    h, w = I.shape
    n = h if axis == 0 else w
    for p in range(2, n):
        table = {}
        ok = True
        for r in range(h):
            for c in range(w):
                if not known[r, c]:
                    continue
                key = (r % p, c) if axis == 0 else (r, c % p)
                v = table.setdefault(key, int(I[r, c]))
                if v != I[r, c]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return None


def _tile(I, known, vp, hp):
    h, w = I.shape
    T = -np.ones((vp, hp), dtype=int)
    for r in range(h):
        for c in range(w):
            if known[r, c]:
                T[r % vp, c % hp] = I[r, c]
    return T


def _infer(I, examples):
    h, w = I.shape
    cands = []
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei); eo = np.asarray(eo)
            if ei.shape == eo.shape:
                d = set(ei[ei != eo].tolist())
                cands += [c for c in d if c not in cands]
    cands += [c for c, _ in Counter(I.flatten().tolist()).most_common()[::-1] if c not in cands]
    for nc in cands:
        if nc not in I:
            continue
        known = I != nc
        vp = _period(I, known, 0)
        hp = _period(I, known, 1)
        if vp is None or hp is None:
            continue
        T = _tile(I, known, vp, hp)
        if (T < 0).any():
            continue
        return nc, vp, hp, T
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    info = _infer(I, examples)
    if info is None:
        return [34], [[0, 0, h - 1, w - 1]]
    nc, vp, hp, T = info
    G = I.copy()

    # The whole grid is ONE tile (vp x hp) repeated from the top-left corner.
    # Tile slots: origins (i*vp, j*hp); slots touching the bottom/right edge are cut short.
    origins = [(r, c) for r in range(0, h, vp) for c in range(0, w, hp)]
    full = [(r, c) for r, c in origins if r + vp <= h and c + hp <= w]

    # 1. Pick the least-damaged full tile slot as the master and mend it from the
    #    tile measured on the intact cells of every other period.
    def dmg(o):
        r, c = o
        return int((G[r:r + vp, c:c + hp] == nc).sum())
    ar, ac = min(full, key=lambda o: (dmg(o), o))
    fix = {}
    for r in range(ar, ar + vp):
        for c in range(ac, ac + hp):
            if G[r, c] == nc:
                fix.setdefault(int(T[r - ar, c - ac]), []).append((r, c))
    for col in sorted(fix):
        ops.append(col); sels.append(sel_of(fix[col]))
        for (r, c) in fix[col]:
            G[r, c] = col
    copy_op = 29 if fix else 28   # master is mended content (CopyO) or untouched input (CopyI)

    # 2. Stamp the master over every other damaged slot, grouped by slot size
    #    (full slots, then edge-cut slots, each with a master cut to that size).
    groups = {}
    for (r, c) in origins:
        if (r, c) == (ar, ac):
            continue
        sh, sw = min(vp, h - r), min(hp, w - c)
        groups.setdefault((sh, sw), []).append((r, c))
    for (sh, sw) in sorted(groups, key=lambda k: (-k[0] * k[1], k)):
        targets = []
        for (r, c) in groups[(sh, sw)]:
            win = G[r:r + sh, c:c + sw]
            src = T[:sh, :sw]
            # paste changes this slot only where it is noise and the tile value is nonzero
            if ((win == nc) & (src != 0)).any():
                targets.append((r, c))
        if not targets:
            continue
        ops.append(copy_op); sels.append([ar, ac, sh - 1, sw - 1])  # rectangle: master tile (cut)
        for (r, c) in targets:
            ops.append(30); sels.append([r, c, 0, 0])
            src = T[:sh, :sw]
            sub = G[r:r + sh, c:c + sw]
            sub[src != 0] = src[src != 0]

    # 3. Paste is transparent for 0: whatever noise is still standing is a 0 cell of the tile.
    rest = [(r, c) for r in range(h) for c in range(w) if G[r, c] == nc]
    if rest:
        ops.append(0); sels.append(sel_of(rest))

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
