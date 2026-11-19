"""
ARC Task: 681b3aeb (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter
import numpy as np


def sample_colors(num_examples=None) -> dict:
    bgc, ca, cb = random.sample(range(10), 3)
    return {"bgc": bgc, "ca": ca, "cb": cb}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None) -> dict:
    from random import randint, choice

    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + round((b - a) * lb)
        hi_ = a + round((b - a) * ub)
        return randint(min(lo, hi_), max(lo, hi_))

    def dneigh(p):
        r, c = p
        return {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}

    def is_rect(S):
        rs = [r for r, _ in S]
        cs = [c for _, c in S]
        return len(S) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

    def normalize(S):
        r0 = min(r for r, _ in S)
        c0 = min(c for _, c in S)
        return {(r - r0, c - c0) for r, c in S}

    if bgc is None:
        bgc, ca, cb = random.sample(range(10), 3)
    for _ in range(1000):
        hi = unifint(diff_lb, diff_ub, (2, 8))
        h_lo, w_lo = 3 * hi, 3 * hi
        if h_lo > max_h or w_lo > max_w:
            continue
        h = unifint(diff_lb, diff_ub, (h_lo, max_h))
        w = unifint(diff_lb, diff_ub, (w_lo, max_w))
        inds = [(r, c) for r in range(hi) for c in range(hi)]
        while True:
            pa = choice(inds)
            pb = choice([p for p in inds if p != pa])
            rem = set(inds) - {pa, pb}
            A, B = {pa}, {pb}
            while rem:
                an = set().union(*[dneigh(p) for p in A]) & rem
                bn = set().union(*[dneigh(p) for p in B]) & rem
                opts = ([0] if an else []) + ([1] if bn else [])
                if choice(opts) == 0:
                    loc = choice(sorted(an)); A.add(loc)
                else:
                    loc = choice(sorted(bn)); B.add(loc)
                rem.discard(loc)
            if not (is_rect(A) and is_rect(B)):
                break
        go = [[ca if (r, c) in A else cb for c in range(hi)] for r in range(hi)]
        An, Bn = normalize(A), normalize(B)
        ha = max(r for r, _ in An) + 1; wa = max(c for _, c in An) + 1
        hb = max(r for r, _ in Bn) + 1; wb = max(c for _, c in Bn) + 1
        ok = False
        for _k in range(10):
            ai = randint(0, h - ha); aj = randint(0, w - wa)
            pA = {(r + ai, c + aj) for r, c in An}
            bi = randint(0, h - hb); bj = randint(0, w - wb)
            pB = {(r + bi, c + bj) for r, c in Bn}
            if pA & pB:
                continue
            # keep the two pieces apart (no 8-adjacency) so each is its own object
            ring = {(r + dr, c + dc) for r, c in pA for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if ring & pB:
                continue
            ok = True
            break
        if not ok:
            continue
        gi = [[bgc] * w for _ in range(h)]
        for r, c in pA:
            gi[r][c] = ca
        for r, c in pB:
            gi[r][c] = cb
        return {"input": gi, "output": go}
    raise RuntimeError("generation failed")


def derive_operations(I, O=None, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # background: the colour the episode's input grids are painted with
    grids = [I]
    for pair in (examples or []):
        try:
            grids.append(np.asarray(pair[0], dtype=int))
        except Exception:
            pass
    votes = Counter()
    for g in grids:
        votes[Counter(g.flatten().tolist()).most_common(1)[0][0]] += 1
    bgc = votes.most_common(1)[0][0]
    if not bool((I == bgc).any()):
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # the two pieces (single-colour 8-connected objects)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
            comps.append((col, sorted(cells)))
    if len(comps) < 2:
        return [34], [[0, 0, h - 1, w - 1]]
    comps.sort(key=lambda t: -len(t[1]))
    comps = comps[:2]
    comps.sort(key=lambda t: t[1][0])

    pieces = []
    for col, cells in comps:
        ul = (min(r for r, _ in cells), min(c for _, c in cells))
        n = {(r - ul[0], c - ul[1]) for r, c in cells}
        pieces.append({"col": col, "cells": cells, "ul": ul, "n": n})
    nA, nB = pieces[0]["n"], pieces[1]["n"]
    span = 2 + max(max(max(r, c) for r, c in nA), max(max(r, c) for r, c in nB))

    # how the pieces interlock into a solid rectangle without overlap
    sol = None
    for dr in range(-span, span + 1):
        for dc in range(-span, span + 1):
            sb = {(r + dr, c + dc) for r, c in nB}
            if sb & nA:
                continue
            u = nA | sb
            r0 = min(r for r, _ in u); c0 = min(c for _, c in u)
            r1 = max(r for r, _ in u); c1 = max(c for _, c in u)
            if len(u) != (r1 - r0 + 1) * (c1 - c0 + 1):
                continue
            sol = (r1 - r0 + 1, c1 - c0 + 1, (-r0, -c0), (dr - r0, dc - c0))
            break
        if sol is not None:
            break
    if sol is None:
        return [34], [[0, 0, h - 1, w - 1]]
    hr, wr = sol[0], sol[1]
    pieces[0]["slot"] = sol[2]
    pieces[1]["slot"] = sol[3]

    def fits(R0, C0):
        return 0 <= R0 and 0 <= C0 and R0 + hr <= h and C0 + wr <= w

    # choose where to assemble: leave one piece in place (anchor) if the
    # rectangle around it fits and the other piece can be copied (non-zero colour)
    plan = None
    for prefer_copyable in (True, False):
        for ai in (0, 1):
            anc, mov = pieces[ai], pieces[1 - ai]
            R0 = anc["ul"][0] - anc["slot"][0]
            C0 = anc["ul"][1] - anc["slot"][1]
            if not fits(R0, C0):
                continue
            if prefer_copyable and mov["col"] == 0:
                continue
            plan = (R0, C0, [mov])
            break
        if plan is not None:
            break
    if plan is None:
        # neither piece can stay: assemble at the nearest in-grid spot to piece A
        a = pieces[0]
        R0 = min(max(a["ul"][0] - a["slot"][0], 0), h - hr)
        C0 = min(max(a["ul"][1] - a["slot"][1], 0), w - wr)
        plan = (R0, C0, [p for p in pieces
                         if (p["ul"][0], p["ul"][1]) != (R0 + p["slot"][0], C0 + p["slot"][1])])
    R0, C0, movers = plan

    ops, sels = [], []
    for p in movers:
        dr_, dc_ = R0 + p["slot"][0], C0 + p["slot"][1]
        if p["col"] != 0:
            # copy the piece (exactly its cells) from the input, stamp it into its slot;
            # since the two pieces tile the rectangle, this covers every non-anchor cell
            ops.append(28); sels.append(sel_of(p["cells"]))
            ops.append(30); sels.append([dr_, dc_, 0, 0])
        else:
            # colour 0 cannot be copied/pasted: paint the piece's shape at its slot
            ops.append(0); sels.append(sel_of([(r + dr_, c + dc_) for r, c in sorted(p["n"])]))
    # keep only the assembled rectangle (selection = that full rectangle)
    ops.append(33); sels.append([R0, C0, hr - 1, wr - 1])
    ops.append(34); sels.append([0, 0, hr - 1, wr - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 681b3aeb"
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
                                f"for task 681b3aeb"
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
                    f"Failed to build a complete episode for task 681b3aeb "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"681b3aeb-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
