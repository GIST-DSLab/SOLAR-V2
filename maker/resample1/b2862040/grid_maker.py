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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a + round(d * (b - a)), a), b)


def _components(G):
    h, w = G.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = G[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and G[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((int(col), cells))
    return comps


def _targets(G, bgc):
    h, w = G.shape
    comps = _components(G)
    enclosed = set()
    for col, cells in comps:
        if col == bgc and not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            enclosed.update(cells)
    res = []
    for col, cells in comps:
        if col == bgc:
            continue
        if any((r + dr, c + dc) in enclosed for r, c in cells
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            res.append(cells)
    return res


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    for _ in range(500):
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        nobjs = _unifint(diff_lb, diff_ub, (1, (h * w) // 16))
        succ, tr, maxtr = 0, 0, 10 * nobjs
        gi = np.full((h, w), bgc, dtype=int)
        inds = {(r, c) for r in range(h) for c in range(w)}
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = random.randint(3, 6)
            ow = random.randint(3, 6)
            obj = {(r, c) for r in range(oh) for c in range(ow)
                   if r in (0, oh - 1) or c in (0, ow - 1)}
            if random.choice((True, False)):
                nkeep = _unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                corners = {(0, 0), (0, ow - 1), (oh - 1, 0), (oh - 1, ow - 1)}
                obj.discard(random.choice(sorted(obj - corners)))
                for _k in range(nrem - 1):
                    xx = [ij for ij in obj
                          if sum((ij[0] + a, ij[1] + b) in obj
                                 for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))) == 1]
                    if not xx:
                        break
                    obj.discard(random.choice(sorted(xx)))
            npert = _unifint(diff_lb, diff_ub, (0, oh + ow))
            rs = [r for r, _ in obj]; cs = [c for _, c in obj]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            objcands = set()
            for k in (1, 2, 3):
                for r in range(r0 - k, r1 + k + 1):
                    for c in range(c0 - k, c1 + k + 1):
                        if r in (r0 - k, r1 + k) or c in (c0 - k, c1 + k):
                            objcands.add((r, c))
            for _k in range(npert):
                nb = {(r + a, c + b) for r, c in obj
                      for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))}
                cand = sorted((objcands - obj) & nb)
                if not cand:
                    break
                obj.add(random.choice(cand))
            mr = min(r for r, _ in obj); mc = min(c for _, c in obj)
            obj = {(r - mr, c - mc) for r, c in obj}
            oh = max(r for r, _ in obj) + 1
            ow = max(c for _, c in obj) + 1
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            lr, lc = random.choice(cands)
            plcd = {(r + lr, c + lc) for r, c in obj}
            if plcd <= inds:
                col = random.choice(remcols)
                for r, c in plcd:
                    gi[r, c] = col
                succ += 1
                ban = set()
                for r, c in plcd:
                    for a in (-1, 0, 1):
                        for b in (-1, 0, 1):
                            ban.add((r + a, c + b))
                inds = inds - ban
        tg = _targets(gi, bgc)
        if tg:
            go = gi.copy()
            for cells in tg:
                for r, c in cells:
                    go[r, c] = 8
            return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    # Background: the canvas colour the generator paints first (dominant colour).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    ops, sels = [], []
    # Every foreground shape touching an enclosed (non-border) background pocket
    # turns 8: one FloodFill8 per shape, seeded on one of its own cells.
    targets = _targets(I, bgc)
    targets.sort(key=lambda cells: min(cells))
    for cells in targets:
        r, c = min(cells)
        ops.append(18)
        sels.append(sel_of([(r, c)]))
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
