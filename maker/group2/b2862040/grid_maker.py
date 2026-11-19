"""
ARC Task: b2862040 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

import numpy as np
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a - 0.5 + (b - a + 1) * d)), b)


def _components4(G):
    """4-connected same-colour components: (comp_id grid, [(colour, cells)])."""
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    cid = -np.ones((h, w), dtype=int)
    comps = []
    for r in range(h):
        for c in range(w):
            if cid[r, c] != -1:
                continue
            col = G[r, c]
            k = len(comps)
            cells = []
            q = deque([(r, c)])
            cid[r, c] = k
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and cid[ny, nx] == -1 and G[ny, nx] == col:
                        cid[ny, nx] = k
                        q.append((ny, nx))
            comps.append((int(col), sorted(cells)))
    return cid, comps


def _closed_object_ids(G, bgc):
    """ids of non-bg components 4-adjacent to an enclosed (non-bordering) bg pocket."""
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    cid, comps = _components4(G)
    targets = set()
    any_pocket = False
    for col, cells in comps:
        if col != bgc:
            continue
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells):
            continue
        any_pocket = True
        for r, c in cells:
            for ny, nx in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= ny < h and 0 <= nx < w and G[ny, nx] != bgc:
                    targets.add(int(cid[ny, nx]))
    return cid, comps, targets, any_pocket


# ----------------------------------------------------------- sample_colors ---
def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ---------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    for _attempt in range(200):
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        h = min(h, max_h)
        w = min(w, max_w)
        nobjs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 16)))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        gi = np.full((h, w), bgc, dtype=int)
        inds = {(r, c) for r in range(h) for c in range(w)}
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6)
            ow = random.randint(3, 6)
            obj = {(r, c) for r in range(oh) for c in range(ow)
                   if r in (0, oh - 1) or c in (0, ow - 1)}
            corners = {(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)}
            if random.choice((True, False)):
                nkeep = _unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                obj.discard(random.choice(sorted(obj - corners)))
                for _ in range(nrem - 1):
                    xx = [ij for ij in obj
                          if len({(ij[0] - 1, ij[1]), (ij[0] + 1, ij[1]),
                                  (ij[0], ij[1] - 1), (ij[0], ij[1] + 1)} & obj) == 1]
                    if not xx:
                        break
                    obj.discard(random.choice(sorted(xx)))
            npert = _unifint(diff_lb, diff_ub, (0, oh + ow))
            rs = [r for r, _ in obj]
            cs = [c for _, c in obj]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            objcands = set()
            for d in (1, 2, 3):
                for r in range(r0 - d, r1 + d + 1):
                    for c in range(c0 - d, c1 + d + 1):
                        if r in (r0 - d, r1 + d) or c in (c0 - d, c1 + d):
                            objcands.add((r, c))
            for _ in range(npert):
                nb = set()
                for (r, c) in obj:
                    nb |= {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
                opts = sorted((objcands - obj) & nb)
                if not opts:
                    break
                obj.add(random.choice(opts))
            mr = min(r for r, _ in obj)
            mc = min(c for _, c in obj)
            obj = {(r - mr, c - mc) for r, c in obj}
            oh = max(r for r, _ in obj) + 1
            ow = max(c for _, c in obj) + 1
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            lr, lc = random.choice(sorted(cands))
            plcd = {(r + lr, c + lc) for r, c in obj}
            if plcd.issubset(inds):
                col = random.choice(remcols)
                for r, c in plcd:
                    gi[r, c] = col
                succ += 1
                ring = set()
                for r, c in plcd:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            ring.add((r + dr, c + dc))
                inds = inds - plcd - ring
        _cid, comps, targets, any_pocket = _closed_object_ids(gi, bgc)
        if any_pocket:
            go = gi.copy()
            for t in targets:
                for r, c in comps[t][1]:
                    go[r, c] = 8
            return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


# ------------------------------------------------------- derive_operations ---
def derive_operations(I, O, examples=None):
    """
    Rule (from I alone): background = dominant colour; every non-background
    4-connected object that touches an enclosed background pocket (a bg region
    not reaching the grid border) - i.e. every CLOSED shape - turns 8.

    Unit of work here: ONE COLOUR AT A TIME.  For each input colour, a single
    Color8 op paints all of that colour's closed shapes together
    (selection = cells of the colour-c objects that enclose a pocket, a set
    nameable from I alone).  O is used only to size the Submit selection.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    _cid, comps, targets, _ = _closed_object_ids(I, bgc)

    by_colour = {}
    for t in targets:
        col, cells = comps[t]
        by_colour.setdefault(col, []).extend(cells)

    ops, sels = [], []
    for col in sorted(by_colour):
        cells = sorted(set(by_colour[col]))
        if not cells:
            continue
        ops.append(8)                     # Color8 on all closed shapes of colour `col`
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b2862040"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b2862040"
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
                                f"for task b2862040"
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
                    f"Failed to build a complete episode for task b2862040 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b2862040-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
