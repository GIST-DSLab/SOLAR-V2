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


def sample_colors(num_examples=None) -> dict:
    bgc, dotc, linc = random.sample(range(10), 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def _unif(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + round(diff_lb * (b - a))
    hi = a + round(diff_ub * (b - a))
    lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])
    HH, WW = (max_w, max_h) if transpose else (max_h, max_w)
    h = _unif(diff_lb, diff_ub, (4, max(4, HH)))
    w = _unif(diff_lb, diff_ub, (4, max(4, WW)))
    gi = np.full((h, w), bgc, dtype=int)
    nl = _unif(diff_lb, diff_ub, (1, max(1, h // 4)))
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
    bad = set(locs) | {l + 1 for l in locs} | {l - 1 for l in locs}
    iopts = [i for i in range(h) if i not in bad]
    jopts = list(range(w))
    ndots = _unif(diff_lb, diff_ub, (1, max(1, min(len(iopts), w // 2))))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        li = random.choice(iopts)
        lj = random.choice(jopts)
        dlocs.append((li, lj))
        jopts = [j for j in jopts if abs(j - lj) > 1]
    go = gi.copy()

    def nb(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                    go[r + dr, c + dc] = linc

    for (li, lj) in dlocs:
        if li < min(locs):
            go[li:min(locs) + 1, lj] = dotc
            nb(min(locs), lj)
        elif li > max(locs):
            go[max(locs):li + 1, lj] = dotc
            nb(max(locs), lj)
        else:
            sp = [e for e in locs if e < li][-1]
            ep = [e for e in locs if e > li][0]
            go[sp:ep + 1, lj] = dotc
            nb(sp, lj)
            nb(ep, lj)
        gi[li, lj] = dotc
    if transpose:
        gi, go = gi.T.copy(), go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    # background dominates (lines <= 1/4 of rows, only a few dots)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # lines are full constant non-bg rows (else columns -> work on the transpose)
    horiz = any(len(set(I[r].tolist())) == 1 and I[r, 0] != bgc for r in range(H))
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(r for r in range(h) if T[r, 0] != bgc and len(set(T[r].tolist())) == 1)
    linc = int(T[locs[0], 0])
    dotc = [c for c in set(T.flatten().tolist()) if c not in (bgc, linc)][0]
    dots = [(r, c) for r in range(h) for c in range(w) if T[r, c] == dotc]

    def mp(r, c):
        return (r, c) if horiz else (c, r)

    lo, hl = locs[0], locs[-1]
    trail, halo = set(), set()
    for (r, c) in dots:
        if r < lo:
            r0, r1, tg = r, lo, [lo]
        elif r > hl:
            r0, r1, tg = hl, r, [hl]
        else:
            r0 = max(e for e in locs if e < r)
            r1 = min(e for e in locs if e > r)
            tg = [r0, r1]
        # the ray: the dot's column segment from the dot to the line(s) it points at
        trail |= {mp(rr, c) for rr in range(r0, r1 + 1)}
        # the ring: 8-neighbourhood of each contact point on a line
        for t in tg:
            halo |= {mp(rr, cc) for rr in range(t - 1, t + 2) for cc in range(c - 1, c + 2)
                     if 0 <= rr < h and 0 <= cc < w and (rr, cc) != (t, c)}
    # 1) every dot shoots its ray (dot colour) to the line(s) in one stroke
    ops = [dotc]; sels = [sel_of(sorted(trail))]
    # 2) every contact point is ringed in line colour (layered over the rays)
    ops.append(linc); sels.append(sel_of(sorted(halo)))
    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
