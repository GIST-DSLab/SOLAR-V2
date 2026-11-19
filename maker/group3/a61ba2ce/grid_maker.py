"""
ARC Task: a61ba2ce (RE-ARC) — LLM-generated grid_maker
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
    # the rule depends only on the background; the four piece colours are free
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        x = random.uniform(lb, ub)
        return int(round(a + x * (b - a)))
    if bgc is None:
        bgc = random.choice(range(10))
    cap_h = max(8, min(30, max_h))
    cap_w = max(8, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (4, min(15, cap_h // 2)))
    w = unifint(diff_lb, diff_ub, (4, min(15, cap_w // 2)))
    lociL = random.randint(2, h - 2)
    lociR = random.randint(2, h - 2)
    locjT = random.randint(2, w - 2)
    locjB = random.randint(2, w - 2)
    c1, c2, c3, c4 = random.sample([c for c in range(10) if c != bgc], 4)

    def line(a, b):
        (r0, c0), (r1, c1_) = a, b
        if r0 == r1:
            return {(r0, c) for c in range(min(c0, c1_), max(c0, c1_) + 1)}
        return {(r, c0) for r in range(min(r0, r1), max(r0, r1) + 1)}
    ulco = line((0, 0), (lociL - 1, 0)) | line((0, 0), (0, locjT - 1))
    urco = line((0, w - 1), (0, locjT)) | line((0, w - 1), (lociR - 1, w - 1))
    llco = line((h - 1, 0), (lociL, 0)) | line((h - 1, 0), (h - 1, locjB - 1))
    lrco = line((h - 1, w - 1), (h - 1, locjB)) | line((h - 1, w - 1), (lociR, w - 1))
    go = np.full((h, w), bgc, dtype=int)
    for o, c in zip((ulco, urco, llco, lrco), (c1, c2, c3, c4)):
        for r, cc in o:
            go[r, cc] = c
    fullh = unifint(diff_lb, diff_ub, (2 * h, cap_h))
    fullw = unifint(diff_lb, diff_ub, (2 * w, cap_w))
    gi = np.full((fullh, fullw), bgc, dtype=int)
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    for _ in range(200):
        free = {(r, c) for r in range(fullh) for c in range(fullw)}
        locs = []
        for o in objs:
            cands = [(i, j) for (i, j) in free
                     if all((r + i, c + j) in free for r, c in o)]
            if not cands:
                break
            loc = random.choice(cands)
            locs.append(loc)
            free -= {(r + loc[0], c + loc[1]) for r, c in o}
        if len(locs) == 4:
            break
    else:
        raise ValueError("placement failed")
    for o, c, (i, j) in zip(objs, ocols, locs):
        for r, cc in o:
            gi[r + i, cc + j] = c
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    fh, fw = I.shape
    ops, sels = [], []

    # background: the canvas colour; four thin L pieces sit on it
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # one colour = one corner piece; its kind is read from the missing bbox corner
    cellmap = {}
    for r in range(fh):
        for c in range(fw):
            v = int(I[r, c])
            if v != bgc:
                cellmap.setdefault(v, []).append((r, c))
    pieces = {}
    for col, cells in cellmap.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        s = set(cells)
        if (r1, c1) not in s:
            kind = 'UL'
        elif (r1, c0) not in s:
            kind = 'UR'
        elif (r0, c1) not in s:
            kind = 'LL'
        else:
            kind = 'LR'
        pieces[kind] = {'color': col, 'cells': sorted(s), 'r0': r0, 'c0': c0,
                        'h': r1 - r0 + 1, 'w': c1 - c0 + 1}
    KINDS = ['UL', 'UR', 'LL', 'LR']
    if not all(k in pieces for k in KINDS):
        ops.append(34); sels.append([0, 0, fh - 1, fw - 1])
        return ops, sels

    h = pieces['UL']['h'] + pieces['LL']['h']
    w = pieces['UL']['w'] + pieces['UR']['w']
    dest = {
        'UL': (0, 0),
        'UR': (0, w - pieces['UR']['w']),
        'LL': (h - pieces['LL']['h'], 0),
        'LR': (h - pieces['LR']['h'], w - pieces['LR']['w']),
    }

    def dest_cells(k):
        p = pieces[k]
        dr, dc = dest[k][0] - p['r0'], dest[k][1] - p['c0']
        return [(r + dr, c + dc) for r, c in p['cells']]

    # 1. open a fresh h x w frame canvas (the whole rectangle -> bbox selection)
    ops.append(33); sels.append([0, 0, h - 1, w - 1])
    canvas = I[:h, :w].copy()   # transparent crop keeps the input's top-left window

    # a piece that already sits in its frame slot inside that window needs nothing
    placed = {k for k in KINDS
              if all(canvas[r, c] == pieces[k]['color'] for r, c in dest_cells(k))}
    keep = set()
    for k in placed:
        keep |= set(dest_cells(k))
    stray = any(canvas[r, c] != bgc and (r, c) not in keep
                for r in range(h) for c in range(w))
    if stray:
        # 2. lay the background base over the whole new canvas
        ops.append(bgc); sels.append([0, 0, h - 1, w - 1])
        placed = set()

    # 3. one colour at a time: copy each piece out of the input, paste it into its corner
    for k in KINDS:
        if k in placed:
            continue
        p = pieces[k]
        if p['color'] == 0:
            # 0 counts as "nothing" for Copy/Paste, so draw this piece's shape directly
            ops.append(0); sels.append(sel_of(dest_cells(k)))
            continue
        ops.append(28); sels.append(sel_of(p['cells']))
        ops.append(30); sels.append([dest[k][0], dest[k][1], 0, 0])

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a61ba2ce"
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
                                f"for task a61ba2ce"
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
                    f"Failed to build a complete episode for task a61ba2ce "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a61ba2ce-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
