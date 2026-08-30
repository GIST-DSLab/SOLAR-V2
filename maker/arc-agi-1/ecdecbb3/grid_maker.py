"""
ARC Task: ecdecbb3 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, dotc, linc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(random.choice(VARIANTS)) for _ in range(n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice((True, False))
    mh, mw = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (4, mh))
    w = unifint(diff_lb, diff_ub, (4, mw))

    gi = [[bgc] * w for _ in range(h)]
    nl = unifint(diff_lb, diff_ub, (1, max(1, h // 4)))
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if len(inds) == 0:
            break
        idx = random.choice(inds)
        locs.append(idx)
        for d in (idx - 2, idx - 1, idx, idx + 1, idx + 2):
            if d in inds:
                inds.remove(d)
    locs = sorted(locs)
    for ii in locs:
        for j in range(w):
            gi[ii][j] = linc

    iopts = [i for i in range(h) if i not in locs and (i - 1) not in locs and (i + 1) not in locs]
    jopts = list(range(w))
    cap = min(len(iopts), w // 2)
    ndots = unifint(diff_lb, diff_ub, (1, cap)) if cap >= 1 else 1

    dlocs = []
    for _ in range(ndots):
        if len(iopts) == 0 or len(jopts) == 0:
            break
        loci = random.choice(iopts)
        locj = random.choice(jopts)
        dlocs.append((loci, locj))
        for d in (locj - 1, locj, locj + 1):
            if d in jopts:
                jopts.remove(d)

    go = [row[:] for row in gi]

    def ring(gr, t, j):
        for r in range(t - 1, t + 2):
            for c in range(j - 1, j + 2):
                if 0 <= r < h and 0 <= c < w and not (r == t and c == j):
                    gr[r][c] = linc

    def seg(gr, r0, r1, j):
        for r in range(min(r0, r1), max(r0, r1) + 1):
            gr[r][j] = dotc

    for (loci, locj) in dlocs:
        if loci < min(locs):
            seg(go, loci, min(locs), locj)
            ring(go, min(locs), locj)
        elif loci > max(locs):
            seg(go, max(locs), loci, locj)
            ring(go, max(locs), locj)
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            seg(go, sp, ep, locj)
            ring(go, sp, locj)
            ring(go, ep, locj)
        gi[loci][locj] = dotc

    if transpose:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {'input': tuple(tuple(r) for r in gi), 'output': tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # full constant non-bg rows -> horizontal lines; else full columns -> vertical
    row_lines = [r for r in range(hi) if I[r, 0] != bgc and len(set(I[r].tolist())) == 1]
    horiz = len(row_lines) > 0
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(r for r in range(h) if T[r, 0] != bgc and len(set(T[r].tolist())) == 1)
    linc = int(T[locs[0], 0])
    others = [c for c in set(T.flatten().tolist()) if c != bgc and c != linc]
    dotc = int(others[0])

    def mp(r, c):
        return (r, c) if horiz else (c, r)

    G = T.copy()
    dots = sorted((r, c) for r in range(h) for c in range(w)
                  if T[r, c] == dotc and r not in locs)

    lo, hi_l = min(locs), max(locs)
    for (dr, dc) in dots:
        if dr < lo:
            r0, r1, targets = dr, lo, [lo]
        elif dr > hi_l:
            r0, r1, targets = hi_l, dr, [hi_l]
        else:
            sp = max(e for e in locs if e < dr)
            ep = min(e for e in locs if e > dr)
            r0, r1, targets = sp, ep, [sp, ep]

        # trail: dot extended to the line(s) it points at
        trail = [(r, dc) for r in range(r0, r1 + 1) if G[r, dc] != dotc]
        if trail:
            ops.append(dotc)
            sels.append(sel_of([mp(r, c) for r, c in trail]))
            for r, c in trail:
                G[r, c] = dotc

        # line-colored halo around each contact point
        for t in targets:
            halo = [(r, c) for r in range(t - 1, t + 2) for c in range(dc - 1, dc + 2)
                    if 0 <= r < h and 0 <= c < w and not (r == t and c == dc)
                    and G[r, c] != linc]
            if halo:
                ops.append(linc)
                sels.append(sel_of([mp(r, c) for r, c in halo]))
                for r, c in halo:
                    G[r, c] = linc

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ecdecbb3"
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
                                f"for task ecdecbb3"
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
                    f"Failed to build a complete episode for task ecdecbb3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ecdecbb3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
