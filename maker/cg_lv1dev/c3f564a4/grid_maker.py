"""
ARC Task: c3f564a4 (RE-ARC) — LLM-generated grid_maker
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
from collections import deque
from maker.sel_helpers import sel_of

# Wallpaper orientation variants: straight stripes vs. diagonal stripes,
# and whether the grid is rotated 90 degrees afterwards.
VARIANTS = [{"diag": d, "rot": r} for d in (False, True) for r in (False, True)]

# Stripe directions to try: lattice key = (a*r + b*c) mod p
_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def sample_colors(num_examples=None) -> dict:
    fixc = random.randrange(10)                       # occluding patch colour
    rem = [c for c in range(10) if c != fixc]
    random.shuffle(rem)                               # wallpaper cycle (first p used)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        ex = [dict(v) for v in VARIANTS]
        ex += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"fixc": fixc, "ccols_pool": rem, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, fixc=0, ccols_pool=None,
             diag=None, rot=None, **kwargs) -> dict:
    from dsl import (interval, canvas, shift, rightmost, mapply, lbind, paint, fill,
                     backdrop, occurrences, dmirror, rot90)
    from utils import unifint
    if ccols_pool is None:
        ccols_pool = [c for c in range(10) if c != fixc]
    if diag is None:
        diag = random.choice((True, False))
    if rot is None:
        rot = random.choice((True, False))
    h = unifint(diff_lb, diff_ub, (7, max_h))
    w = unifint(diff_lb, diff_ub, (7, max_w))
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h // 3, w // 3))))
    ccols = list(ccols_pool[:p])
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {x for x in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if not diag:
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = random.randint(0, h - oh)
        locj = random.randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if (len(occurrences(tmpg, baseobj)) > 1
                and len([r for r in tmpg if fixc not in r]) > 0
                and len([r for r in dmirror(tmpg) if fixc not in r]) > 0):
            gi = tmpg
            succ += 1
        tr += 1
    if rot:
        gi = rot90(gi)
        go = rot90(go)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading (shared with the golf entry): the whole grid is ONE cyclic colour
    # sequence laid out as stripes. Each stripe family (lattice class) carries a
    # single wallpaper colour. So the unit of work is a colour's stripe family,
    # not an occluding patch: walk the colour cycle and, for each colour, close
    # every gap on that colour's stripes in one Color op. (The maker instead
    # works patch by patch, then colour by colour inside each patch.)
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Occluding colour, taken from the demonstrations: the colour that appears in
    # every example input and never survives into the matching output.
    noise_from_ex = None
    if examples:
        cand = None
        for ei, eo in examples:
            ei = np.asarray(ei)
            eo = np.asarray(eo)
            gone = set(ei.flatten().tolist()) - set(eo.flatten().tolist())
            cand = gone if cand is None else (cand & gone)
        if cand and len(cand) == 1:
            noise_from_ex = cand.pop()

    # Lines with the shortest period have no patches on them, so their colours
    # are wallpaper colours. Used to tell wallpaper from patch colour.
    def line_period(line):
        n = len(line)
        for q in range(1, n):
            if all(line[c] == line[c + q] for c in range(n - q)):
                return q
        return n

    rows = [I[r].tolist() for r in range(hi)]
    colsl = [I[:, c].tolist() for c in range(wi)]
    rp = [line_period(x) for x in rows]
    cp = [line_period(x) for x in colsl]
    pattern_cols = set()
    for r in range(hi):
        if rp[r] == min(rp):
            pattern_cols |= set(rows[r])
    for c in range(wi):
        if cp[c] == min(cp):
            pattern_cols |= set(colsl[c])

    palette = sorted(set(I.flatten().tolist()))
    noises = [noise_from_ex] if noise_from_ex in palette else palette
    R, C = np.indices((hi, wi))
    cands = []
    for (a, b) in _DIRS:
        for p in range(2, 10):
            key = (a * R + b * C) % p
            for k in noises:
                keep = I != k
                mapping = {}
                bad = False
                for cls in range(p):
                    u = np.unique(I[(key == cls) & keep])
                    if len(u) > 1:
                        bad = True
                        break
                    if len(u) == 1:
                        mapping[cls] = int(u[0])
                if bad or not mapping or len(set(mapping.values())) != len(mapping):
                    continue
                if len(np.unique(key[I == k])) < 2 and noise_from_ex is None:
                    continue
                cands.append((k not in pattern_cols, len(mapping) == p, int(keep.sum()), -p,
                              k, (a, b), p, mapping))
    cands.sort(reverse=True, key=lambda t: t[:4])

    if cands:
        _, _, _, _, noise, (a, b), p, mapping = cands[0]
        key = (a * R + b * C) % p
        # Go through the colour cycle in order. For each colour, select the
        # patch-coloured cells on that colour's stripes and paint them that colour.
        for cls in range(p):
            v = mapping.get(cls)
            if v is None or v == noise:
                continue
            cells = [(int(r), int(c)) for r, c in zip(*np.nonzero((key == cls) & (I == noise)))]
            if cells:
                ops.append(int(v))
                sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c3f564a4"
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
                                f"for task c3f564a4"
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
                    f"Failed to build a complete episode for task c3f564a4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c3f564a4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
