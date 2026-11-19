"""
ARC Task: ddf7fa4f (RE-ARC) — LLM-generated grid_maker
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
from collections import deque, Counter
from maker.sel_helpers import sel_of

# The golf entry and the maker read this task the same way, apart from one detail.
# Each block takes the colour of the marker that sits on the marker edge in line with it
# (same column for a top or bottom edge, same row for a left or right edge), and each block
# is recoloured as one unit.
# The golf entry looks only along that one axis. The maker also accepts a marker that shares
# the other axis, which the generator never needs. Here only that one axis is used.
# Two other changes:
# - The marker edge is found from the strip of background the generator always leaves next
#   to it.
# - Markers are handled in their order along that edge.
# A later block can cover the column of an earlier block's marker. So a block is settled only
# when exactly one marker left in play lines up with it. Settled markers are then removed,
# just as in the maker.
# Each block is recoloured with one FloodFill. Blocks never touch each other side by side,
# so the fill cannot spread into a neighbour.

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS] + [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None, **kw):
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + d * (b - a)))))
    if rot is None:
        rot = random.choice(ROTS)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    if rot in (1, 3):
        h, w = min(h, max_w), min(w, max_h)
    nocc = unifint(diff_lb, diff_ub, (1, min(w // 3, (h * w) // 36)))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    succ, tr, maxtr = 0, 0, 10 * nocc
    inds = {(i, j) for i in range(2, h) for j in range(w)}
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        has = {j for j in range(w) if gi[0, j] == bgc}
        cant = {j for j in range(w) if gi[0, j] != bgc}
        jopts = [j for j in range(w)
                 if set(range(j, j + ow)) & has and not (set(range(j, j + ow)) & cant)]
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] in jopts]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        locat = random.choice([jj for jj in range(locj, locj + ow) if jj in has])
        sq = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if sq <= inds:
            succ += 1
            nb = {(i + a, j + b) for i, j in sq for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))}
            inds = inds - sq - nb
            col = random.choice(remcols)
            gr = random.choice([c for c in remcols if c != col])
            gi[0, locat] = col
            go[0, locat] = col
            for i, j in sq:
                gi[i, j] = gr
                go[i, j] = col
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    border = list(I[0]) + list(I[-1]) + list(I[:, 0]) + list(I[:, -1])
    bgc = int(Counter(int(v) for v in border).most_common(1)[0][0])

    # all 4-connected single-colour components
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            comp = []
            dq = deque([(r, c)])
            seen[r, c] = True
            while dq:
                a, b = dq.popleft()
                comp.append((a, b))
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        dq.append((na, nb))
            comps.append(comp)

    # candidate marker edges: (edge cells, the parallel line just inside it, projection axis)
    # axis 'c': edge is a row -> a block lines up with a marker by sharing a column
    cands = [
        ([(0, c) for c in range(w)], [(1, c) for c in range(w)], 'c'),
        ([(h - 1, c) for c in range(w)], [(h - 2, c) for c in range(w)], 'c'),
        ([(r, 0) for r in range(h)], [(r, 1) for r in range(h)], 'r'),
        ([(r, w - 1) for r in range(h)], [(r, w - 2) for r in range(h)], 'r'),
    ]
    plan = None
    for line, inner, ax in cands:
        if any(I[p] != bgc for p in inner):
            continue                       # the marker edge always has an empty strip beside it
        markers = [p for p in line if I[p] != bgc]
        if not markers:
            continue
        ls = set(line)
        blocks = [cp for cp in comps if not (set(cp) & ls)]
        k = 1 if ax == 'c' else 0
        spans = [{p[k] for p in b} for b in blocks]
        rem_b = list(range(len(blocks)))
        rem_m = list(markers)
        pairs = []
        while rem_b:
            rnd = []
            for m in rem_m:                # sweep along the marker edge
                hits = [i for i in rem_b if m[k] in spans[i]]
                for i in hits:
                    if sum(1 for mm in rem_m if mm[k] in spans[i]) == 1:
                        rnd.append((m, i))
            if not rnd:
                break
            for m, i in rnd:
                if m not in rem_m or i not in rem_b:
                    continue
                pairs.append((m, blocks[i]))
                rem_b.remove(i)
                rem_m.remove(m)
        if not rem_b:
            plan = pairs
            break

    ops, sels = [], []
    for m, b in plan or []:
        newc = int(I[m])
        old = int(I[b[0]])
        if newc == old:
            continue
        seed = min(b)
        ops.append(10 + newc)              # flood the whole block with its marker's colour
        sels.append([seed[0], seed[1], 0, 0])
    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
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
                                f"for task ddf7fa4f"
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
                    f"Failed to build a complete episode for task ddf7fa4f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ddf7fa4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
