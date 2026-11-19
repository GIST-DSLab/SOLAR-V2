"""
ARC Task: e8593010 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    # 1/2/3 are the size labels the output writes, so the canvas and the blobs
    # are drawn from the rest of the palette (as the original generator does)
    cols = [c for c in range(10) if c not in (1, 2, 3)]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc, fgc) -> dict:
    a = frozenset({frozenset({ORIGIN})})
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
        frozenset({ORIGIN, DOWN, UNITY}),
        frozenset({ORIGIN, DOWN, RIGHT}),
        frozenset({UNITY, DOWN, RIGHT}),
        frozenset({UNITY, ORIGIN, RIGHT}),
        shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
        shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    a, b, c = totuple(a), totuple(b), totuple(c)
    prs = [(a, 3), (b, 2), (c, 1)]

    hlo = min(5, max_h)
    wlo = min(5, max_w)
    h = unifint(diff_lb, diff_ub, (hlo, max_h))
    w = unifint(diff_lb, diff_ub, (wlo, max_w))

    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    cap = max(1, ((h * w) // 2) // 2)
    # a handful of blobs at minimum, so every instance exhibits all three sizes
    nobjs = unifint(diff_lb, diff_ub, (min(6, cap), cap))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        # the first three placements cover size 1 / 2 / 3, so the size->colour
        # rule is always demonstrated; afterwards the uniform choice resumes
        if k < len(prs):
            objs, col = prs[k]
        else:
            objs, col = choice(prs)
        obj = choice(objs)
        while ntr < maxtr:
            if len(reminds) == 0:
                break
            loc = choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Blobs of the same shape all end up identical in the output, so the work is
    done once per shape: flood one blob of that shape to the colour its size
    calls for, then stamp that finished blob (CopyO + Paste) onto its repeats."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # ── background / blob colour, and the blobs themselves ───────────────────
    def components(grid, colour):
        gh, gw = grid.shape
        seen = set()
        out = []
        for r in range(gh):
            for c in range(gw):
                if grid[r, c] == colour and (r, c) not in seen:
                    stack = [(r, c)]
                    seen.add((r, c))
                    cur = []
                    while stack:
                        rr, cc = stack.pop()
                        cur.append((rr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = rr + dr, cc + dc
                            if (0 <= nr < gh and 0 <= nc < gw
                                    and grid[nr, nc] == colour
                                    and (nr, nc) not in seen):
                                seen.add((nr, nc))
                                stack.append((nr, nc))
                    out.append(sorted(cur))
        return out

    def split_bg_fg(grid):
        cnt = Counter(grid.flatten().tolist())
        colours = sorted(cnt)
        if len(colours) == 1:
            return colours[0], None
        # the blobs are tiny (1..3 cells); the canvas colour is not
        tiny = [col for col in colours
                if max(len(x) for x in components(grid, col)) <= 3]
        if len(tiny) == 1:
            rest = [col for col in colours if col != tiny[0]]
            bg = rest[0] if len(rest) == 1 else max(rest, key=lambda x: cnt[x])
            return bg, tiny[0]
        bg = max(colours, key=lambda x: cnt[x])
        rest = [col for col in colours if col != bg]
        return bg, (rest[0] if rest else None)

    bgc, fgc = split_bg_fg(I)
    ops, sels = [], []
    if fgc is None:
        ops.append(34)
        sels.append([0, 0, h - 1, w - 1])
        return ops, sels

    # ── size -> colour convention, read off the demonstrations ───────────────
    size2col = {}
    for pairs, from_examples in ((examples or [], True), ([(I, O)], False)):
        for ei, eo in pairs:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            _, efg = split_bg_fg(ei)
            if efg is None:
                continue
            for cells in components(ei, efg):
                r0, c0 = cells[0]
                size2col.setdefault(len(cells), int(eo[r0, c0]))
        if from_examples and size2col:
            break            # the examples fix the convention; never consult O

    objs = components(I, fgc)

    # ── group the blobs by shape: same shape => same finished stamp ──────────
    groups = {}
    for cells in objs:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        key = tuple(sorted((r - r0, c - c0) for r, c in cells))
        groups.setdefault(key, []).append((r0, c0, cells))

    # shape by shape, size class by size class (dots, then dominoes, then
    # trominoes); inside a shape, its blobs top-to-bottom / left-to-right
    work = np.array(I)
    for key in sorted(groups, key=lambda k: (len(k), k)):
        members = sorted(groups[key], key=lambda mb: (mb[0], mb[1]))
        newcol = size2col.get(len(key))
        if newcol is None or newcol == fgc:
            continue
        ecells = members[0][2]
        seed = ecells[0]

        # 1. recolour one blob of this shape whole, from a seed inside it
        region = [x for x in components(work, int(work[seed])) if seed in x][0]
        if set(region) == set(ecells):
            ops.append(10 + newcol)                 # FloodFill<newcol>
            sels.append([seed[0], seed[1], 0, 0])
        else:                                       # defensive: fill would leak
            ops.append(newcol)                      # Color<newcol>
            sels.append(sel_of(ecells))
        for (r, c) in ecells:
            work[r, c] = newcol

        # 2. that finished blob IS the stamp for every repeat of the shape
        if len(members) > 1:
            ops.append(29)                          # CopyO — the recoloured blob
            sels.append(sel_of(ecells))             # its cells only, no bbox filler
            for (mr, mc, mcells) in members[1:]:
                ops.append(30)                      # Paste it over this repeat
                sels.append([mr, mc, 0, 0])
                for (r, c) in mcells:
                    work[r, c] = newcol

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
                        f"num_examples+1 ({num_examples + 1}) for task e8593010"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e8593010"
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
                                f"for task e8593010"
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
                    f"Failed to build a complete episode for task e8593010 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e8593010-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
