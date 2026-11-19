"""
ARC Task: 54d9e175 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
import random
from collections import Counter

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - harness provides sel_of
    def sel_of(cells):
        return {"cells": [(int(r), int(c)) for r, c in cells]}


# ----------------------------------------------------------------------------
# 1. colors
# ----------------------------------------------------------------------------
def sample_colors(num_examples=None) -> dict:
    # generator does: linc, bgc = sample((0, 5), 2)
    pair = [0, 5]
    random.shuffle(pair)
    linc, bgc = pair[0], pair[1]
    return {"linc": linc, "bgc": bgc}


# ----------------------------------------------------------------------------
# 2. generator
# ----------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, linc, bgc) -> dict:
    h = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_h - 1))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(5, max_w - 1))))

    nh_ub = max(1, (max_h + 1) // (h + 1))
    nw_ub = max(1, (max_w + 1) // (w + 1))

    nh = unifint(diff_lb, diff_ub, (1, nh_ub))
    nw_lb = 1 if nh > 1 else 2
    if nw_ub < nw_lb:
        if nh_ub >= 2:
            nh = unifint(diff_lb, diff_ub, (2, nh_ub))
            nw_lb = 1
        else:
            nw_lb = min(nw_lb, nw_ub)
    nw = unifint(diff_lb, diff_ub, (nw_lb, max(nw_lb, nw_ub)))

    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1

    gi = canvas(linc, (fullh, fullw))
    go = canvas(linc, (fullh, fullw))
    obj = asindices(canvas(bgc, (h, w)))
    for a in range(nh):
        for b in range(nw):
            plcd = shift(obj, (a * (h + 1), b * (w + 1)))
            icol = randint(1, 4)
            ocol = icol + 5
            gi = fill(gi, bgc, plcd)
            go = fill(go, ocol, plcd)
            dot = choice(totuple(plcd))
            gi = fill(gi, icol, {dot})
    return {'input': gi, 'output': go}


# ----------------------------------------------------------------------------
# 3. operations
# ----------------------------------------------------------------------------
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    MARKERS = (1, 2, 3, 4)

    # ---- structure reading (input only) ------------------------------------
    def comps_for(G, L):
        gh, gw = G.shape
        seen = np.zeros((gh, gw), dtype=bool)
        out = []
        for r in range(gh):
            for c in range(gw):
                if G[r, c] == L or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < gh and 0 <= nx < gw and not seen[ny, nx] and G[ny, nx] != L:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                out.append(cells)
        return out

    def analyze(G):
        """Find the separator colour and the marked cells (blocks) of G."""
        G = np.asarray(G, dtype=int)
        gh, gw = G.shape
        nmk = int(sum(1 for r in range(gh) for c in range(gw) if int(G[r, c]) in MARKERS))

        def build(L, strict):
            other = 5 if L == 0 else 0
            blocks = []
            shapes = set()
            for cells in comps_for(G, L):
                rs = [y for y, _ in cells]
                cs = [x for _, x in cells]
                r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
                bh, bw = r1 - r0 + 1, c1 - c0 + 1
                mk = [int(G[y, x]) for y, x in cells if int(G[y, x]) in MARKERS]
                if strict:
                    if bh * bw != len(cells) or len(mk) != 1:
                        return None
                    if any(int(G[y, x]) not in MARKERS and int(G[y, x]) != other for y, x in cells):
                        return None
                elif len(mk) != 1:
                    continue
                shapes.add((bh, bw))
                blocks.append({'r': r0, 'c': c0, 'h': bh, 'w': bw,
                               'm': mk[0], 'cells': sorted(cells)})
            if strict and (len(blocks) != nmk or len(shapes) != 1 or not blocks):
                return None
            if not blocks:
                return None
            return blocks

        for L in (0, 5):
            got = build(L, True)
            if got is not None:
                return got
        # lenient fallback: densest colour is the separator
        L = Counter(np.asarray(G).flatten().tolist()).most_common(1)[0][0]
        return build(L, False)

    blocks = analyze(I)
    if not blocks:
        return [34], [[0, 0, hi - 1, wi - 1]]

    # ---- which colour a marked cell becomes (read from the demonstrations) --
    mapping = {}
    if examples:
        tally = {}
        for pair in examples:
            try:
                Ie, Oe = pair[0], pair[1]
            except Exception:
                continue
            bl = analyze(Ie)
            if not bl:
                continue
            Oe = np.asarray(Oe, dtype=int)
            for b in bl:
                oc = int(Oe[b['r'], b['c']])
                tally.setdefault(b['m'], Counter())[oc] += 1
        for m, cc in tally.items():
            mapping[m] = int(cc.most_common(1)[0][0])

    def outcol(m):
        return int(mapping.get(m, m + 5))

    # ---- lay the blocks out on their own little grid ------------------------
    rws = sorted(set(b['r'] for b in blocks))
    cls = sorted(set(b['c'] for b in blocks))
    for b in blocks:
        b['i'] = rws.index(b['r'])
        b['j'] = cls.index(b['c'])
        b['o'] = outcol(b['m'])
    nR, nC = len(rws), len(cls)
    n = len(blocks)

    # ---- route, decided by what the input itself looks like -----------------
    msum = sum(b['m'] for b in blocks)
    strategy = (msum + n) % 4
    first_mark = min(blocks, key=lambda b: (b['i'], b['j']))['m']
    order_k = (first_mark + nC) % 4

    def okey(b):
        if order_k == 0:                      # rows, left to right
            return (b['i'], b['j'])
        if order_k == 1:                      # rows, snaking back and forth
            return (b['i'], b['j'] if b['i'] % 2 == 0 else -b['j'])
        if order_k == 2:                      # columns, top to bottom
            return (b['j'], b['i'])
        return (-b['i'], -b['j'])             # rows, bottom-right first

    seq = sorted(blocks, key=okey)

    def groups_in_order(items):
        g = {}
        for b in items:
            g.setdefault(b['o'], []).append(b)
        return list(g.items())   # dicts keep first-seen order

    ops, sels = [], []

    if strategy == 0:
        # one marked cell at a time, along the chosen sweep
        for b in seq:
            ops.append(b['o'])
            sels.append(sel_of(b['cells']))

    elif strategy == 1:
        # lay the commonest colour across every cell, then correct the others
        cnt = Counter(b['o'] for b in seq)
        top = max(cnt.values())
        base = min(c for c, k in cnt.items() if k == top)
        base_cells = []
        for b in seq:
            base_cells.extend(b['cells'])
        ops.append(int(base))
        sels.append(sel_of(base_cells))
        for b in seq:
            if b['o'] != base:
                ops.append(b['o'])
                sels.append(sel_of(b['cells']))

    elif strategy == 2:
        # all cells sharing a marker get painted together, colour by colour
        for col, bs in groups_in_order(seq):
            cells = []
            for b in bs:
                cells.extend(b['cells'])
            ops.append(int(col))
            sels.append(sel_of(cells))

    else:
        # paint one of each colour, then copy that finished cell onto its twins
        for col, bs in groups_in_order(seq):
            src = bs[0]
            ops.append(int(col))
            sels.append(sel_of(src['cells']))
            if len(bs) > 1:
                # rectangular region = the whole block, so a bbox is exact here
                ops.append(29)
                sels.append([src['r'], src['c'], src['h'] - 1, src['w'] - 1])
                for b in bs[1:]:
                    ops.append(30)
                    sels.append([b['r'], b['c'], 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 54d9e175"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 54d9e175"
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
                                f"for task 54d9e175"
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
                    f"Failed to build a complete episode for task 54d9e175 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"54d9e175-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
