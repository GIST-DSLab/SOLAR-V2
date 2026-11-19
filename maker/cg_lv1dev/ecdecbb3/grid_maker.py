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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b <= a:
        return a
    d = random.uniform(diff_lb, diff_ub)
    lo = a + int(round((b - a) * max(0.0, d - 0.1)))
    hi_ = a + int(round((b - a) * min(1.0, d + 0.1)))
    lo, hi_ = max(a, min(lo, b)), max(a, min(hi_, b))
    if hi_ < lo:
        lo, hi_ = hi_, lo
    return random.randint(lo, hi_)


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, linc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants]
        ex += [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    HH, WW = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unifint(diff_lb, diff_ub, (4, max(4, HH)))
    w = _unifint(diff_lb, diff_ub, (4, max(4, WW)))
    gi = np.full((h, w), bgc, dtype=int)
    nl = _unifint(diff_lb, diff_ub, (1, h // 4))
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if not inds:
            break
        idx = random.choice(inds)
        locs.append(idx)
        inds = [i for i in inds if abs(i - idx) > 2]
    locs = sorted(locs)
    for ii in locs:
        gi[ii, :] = linc
    excl = set(locs) | {l + 1 for l in locs} | {l - 1 for l in locs}
    iopts = [i for i in range(h) if i not in excl]
    jopts = list(range(w))
    ndots = _unifint(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        loci = random.choice(iopts)
        locj = random.choice(jopts)
        dlocs.append((loci, locj))
        jopts = [j for j in jopts if abs(j - locj) > 1]
    if not dlocs:
        raise ValueError("no dots")
    go = gi.copy()

    def nb(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                    go[r + dr, c + dc] = linc

    for (loci, locj) in dlocs:
        if loci < min(locs):
            go[loci:min(locs) + 1, locj] = dotc
            nb(min(locs), locj)
        elif loci > max(locs):
            go[max(locs):loci + 1, locj] = dotc
            nb(max(locs), locj)
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            go[sp:ep + 1, locj] = dotc
            nb(sp, locj)
            nb(ep, locj)
        gi[loci, locj] = dotc
    if transpose:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # background: the colour the canvas is painted with (lines <= 1/4, dots sparse)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Lines are full constant non-bg rows (else full columns). Work in a frame
    # where lines are horizontal; mp() maps back to grid coordinates.
    horiz = any(I[r, 0] != bgc and len(set(I[r].tolist())) == 1 for r in range(hi))
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(r for r in range(h) if T[r, 0] != bgc and len(set(T[r].tolist())) == 1)
    linc = int(T[locs[0], 0])
    dots = sorted((r, c) for r in range(h) for c in range(w)
                  if r not in locs and T[r, c] != bgc)
    dotc = int(T[dots[0]])

    def mp(cells):
        return [(r, c) if horiz else (c, r) for r, c in cells]

    G = T.copy()
    lo, hl = locs[0], locs[-1]

    # Phase 1 -- rays. Each dot shoots a ray toward every line it faces
    # (one line if it sits outside the band, both neighbours if between two
    # lines). Each ray is one unit: the cells from the dot up to and
    # including the line cell it strikes.
    contacts = []   # (line row, column) where a ray pierced a line
    for (dr, dc) in dots:
        if dr < lo:
            targets = [lo]
        elif dr > hl:
            targets = [hl]
        else:
            targets = [max(e for e in locs if e < dr), min(e for e in locs if e > dr)]
        for t in targets:
            step = 1 if t > dr else -1
            ray = [(r, dc) for r in range(dr + step, t + step, step)]
            ops.append(dotc); sels.append(sel_of(mp(ray)))
            for cell in ray:
                G[cell] = dotc
            contacts.append((t, dc))

    # Phase 2 -- crossings. Line by line, every point a ray pierced gets a
    # line-coloured ring around it (the 8 neighbours not already line colour).
    for t in locs:
        for (tr, dc) in sorted(c for c in contacts if c[0] == t):
            ring = [(r, c) for r in range(tr - 1, tr + 2) for c in range(dc - 1, dc + 2)
                    if 0 <= r < h and 0 <= c < w and (r, c) != (tr, dc)
                    and G[r, c] != linc]
            if ring:
                ops.append(linc); sels.append(sel_of(mp(ring)))
                for cell in ring:
                    G[cell] = linc

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
