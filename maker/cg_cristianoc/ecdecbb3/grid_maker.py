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
    n = num_examples if num_examples else 3
    # discrete variant: horizontal rails vs. vertical rails (dmirror)
    if n >= 2:
        ex = [{"mirror": False}, {"mirror": True}]
    else:
        ex = [{"mirror": random.choice([False, True])}]
    ex += [{"mirror": random.choice([False, True])} for _ in range(n - len(ex))]
    random.shuffle(ex)
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, dotc, linc, mirror=None) -> dict:
    def unifint(lo, hi):
        lo2 = lo + (hi - lo) * diff_lb
        hi2 = lo + (hi - lo) * diff_ub
        return max(lo, min(hi, int(round(random.uniform(lo2, hi2)))))

    if mirror is None:
        mirror = random.choice([False, True])
    H, W = (max_w, max_h) if mirror else (max_h, max_w)
    h = unifint(4, max(4, min(30, H)))
    w = unifint(4, max(4, min(30, W)))
    gi = np.full((h, w), bgc, dtype=int)
    nl = unifint(1, max(1, h // 4))
    inds = list(range(h))
    locs = []
    for _ in range(nl):
        if not inds:
            break
        idx = random.choice(inds)
        locs.append(idx)
        inds = [x for x in inds if abs(x - idx) > 2]
    locs = sorted(locs)
    for ii in locs:
        gi[ii, :] = linc
    iopts = [i for i in range(h) if all(abs(i - l) > 1 for l in locs)]
    jopts = list(range(w))
    ndots = unifint(1, max(1, min(len(iopts), w // 2)))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        a = random.choice(iopts)
        b = random.choice(jopts)
        dlocs.append((a, b))
        jopts = [j for j in jopts if abs(j - b) > 1]
    go = gi.copy()

    def nb(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                    go[r + dr, c + dc] = linc

    for (a, b) in dlocs:
        if a < min(locs):
            go[a:min(locs) + 1, b] = dotc
            nb(min(locs), b)
        elif a > max(locs):
            go[max(locs):a + 1, b] = dotc
            nb(max(locs), b)
        else:
            sp = max(e for e in locs if e < a)
            ep = min(e for e in locs if e > a)
            go[sp:ep + 1, b] = dotc
            nb(sp, b)
            nb(ep, b)
        gi[a, b] = dotc
    if mirror:
        gi, go = gi.T, go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # Reading (from the code-golf solver): each dot "lands" on the rail(s) it points at,
    # stamping a 3x3 line-coloured SQUARE centred on the contact point; then the dot's ray
    # is drawn up to the rim of that square and its centre is marked with the dot colour.
    # Unit = per dot: square(s) first, then the ray + centre(s).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]  # rails <= h/4 rows, dots sparse

    def full_lines(A):
        return [r for r in range(A.shape[0])
                if A[r, 0] != bgc and len(set(A[r].tolist())) == 1]

    horiz = len(full_lines(I)) > 0
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(full_lines(T))
    linc = int(T[locs[0], 0])
    dotc = [c for c in set(T.flatten().tolist()) if c not in (bgc, linc)][0]
    mp = (lambda r, c: (r, c)) if horiz else (lambda r, c: (c, r))

    G = T.copy()
    ops, sels = [], []
    dots = sorted((r, c) for r in range(h) for c in range(w)
                  if T[r, c] == dotc and r not in locs)
    lo, hl = locs[0], locs[-1]
    for (dr, dc) in dots:
        # rails this dot points at: the near rail when outside, both bounding rails when between
        if dr < lo:
            contacts = [lo]
        elif dr > hl:
            contacts = [hl]
        else:
            contacts = [max(e for e in locs if e < dr), min(e for e in locs if e > dr)]
        # 1) stamp a 3x3 line-coloured square on each contact point (clipped to grid)
        for t in contacts:
            sq = [(r, c) for r in range(t - 1, t + 2) for c in range(dc - 1, dc + 2)
                  if 0 <= r < h and 0 <= c < w and G[r, c] != linc]
            if sq:
                ops.append(linc)
                sels.append(sel_of([mp(r, c) for r, c in sq]))
                for r, c in sq:
                    G[r, c] = linc
        # 2) ray from the dot to the squares' rims, plus each square's centre
        a, b = min([dr] + contacts), max([dr] + contacts)
        ray = [(r, dc) for r in range(a, b + 1) if all(abs(r - t) >= 2 for t in contacts)]
        ray += [(t, dc) for t in contacts]
        ray = [(r, c) for r, c in ray if G[r, c] != dotc]
        if ray:
            ops.append(dotc)
            sels.append(sel_of([mp(r, c) for r, c in ray]))
            for r, c in ray:
                G[r, c] = dotc

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
