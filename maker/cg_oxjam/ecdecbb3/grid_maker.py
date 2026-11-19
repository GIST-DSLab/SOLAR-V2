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
    variants = [{"mirrored": False}, {"mirrored": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in variants] + [dict(random.choice(variants)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(variants))]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "dotc": dotc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=0, dotc=2, linc=8, mirrored=None) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])

    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))

    H, W = (max_w, max_h) if mirrored else (max_h, max_w)
    h = unif(4, min(30, H))
    w = unif(4, min(30, W))
    gi = [[bgc] * w for _ in range(h)]
    nl = unif(1, h // 4)
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
        gi[ii] = [linc] * w
    bad = set(locs) | {l + 1 for l in locs} | {l - 1 for l in locs}
    iopts = [i for i in range(h) if i not in bad]
    jopts = list(range(w))
    if not iopts:
        raise ValueError("no dot rows")
    ndots = unif(1, min(len(iopts), w // 2))
    dlocs = []
    for _ in range(ndots):
        if not iopts or not jopts:
            break
        li = random.choice(iopts)
        lj = random.choice(jopts)
        dlocs.append((li, lj))
        jopts = [x for x in jopts if abs(x - lj) > 1]
    go = [row[:] for row in gi]

    def nb(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                    go[r + dr][c + dc] = linc

    for (li, lj) in dlocs:
        if li < locs[0]:
            for r in range(li, locs[0] + 1):
                go[r][lj] = dotc
            nb(locs[0], lj)
        elif li > locs[-1]:
            for r in range(locs[-1], li + 1):
                go[r][lj] = dotc
            nb(locs[-1], lj)
        else:
            sp = [e for e in locs if e < li][-1]
            ep = [e for e in locs if e > li][0]
            for r in range(sp, ep + 1):
                go[r][lj] = dotc
            nb(sp, lj)
            nb(ep, lj)
        gi[li][lj] = dotc
    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (per-direction, as in the golf solution): each dot fires a ray in each of
    # the two directions perpendicular to the lines; a ray exists only if a line lies
    # that way, and it stops ON the nearest line. Work one direction at a time
    # (towards lower index, then towards higher index): draw every ray of that
    # direction, then ring each contact point of that direction with the line colour.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []

    def uniform_lines(G):
        return [r for r in range(G.shape[0])
                if G[r, 0] != bgc and len(set(G[r].tolist())) == 1]

    horiz = len(uniform_lines(I)) > 0
    T = I if horiz else I.T
    h, w = T.shape
    locs = sorted(uniform_lines(T))
    if locs:
        linc = int(T[locs[0], 0])
        others = sorted(c for c in set(T.flatten().tolist()) if c not in (bgc, linc))
        mp = (lambda r, c: (r, c)) if horiz else (lambda r, c: (c, r))
        if others:
            dotc = int(others[0])
            G = T.copy()
            dots = sorted((r, c) for r in range(h) for c in range(w)
                          if T[r, c] == dotc and r not in locs)
            for step in (-1, 1):
                contacts = []
                # rays of this direction: from the dot up to and including the nearest line
                for (dr, dc) in dots:
                    ahead = [l for l in locs if (l - dr) * step > 0]
                    if not ahead:
                        continue
                    t = min(ahead, key=lambda l: abs(l - dr))
                    ray = [(r, dc) for r in range(dr + step, t + step, step)
                           if G[r, dc] != dotc]
                    if ray:
                        ops.append(dotc)
                        sels.append(sel_of([mp(r, c) for r, c in ray]))
                        for r, c in ray:
                            G[r, c] = dotc
                    contacts.append((t, dc))
                # line-coloured ring around every contact point of this direction
                for (t, dc) in contacts:
                    ring = [(r, c) for r in range(t - 1, t + 2) for c in range(dc - 1, dc + 2)
                            if 0 <= r < h and 0 <= c < w and (r, c) != (t, dc)
                            and G[r, c] != linc]
                    if ring:
                        ops.append(linc)
                        sels.append(sel_of([mp(r, c) for r, c in ring]))
                        for r, c in ring:
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
