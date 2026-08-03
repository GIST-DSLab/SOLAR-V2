"""
ARC Task: 150deff5 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 8)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}

    def colorings(cells):
        """all distinct cell->color maps that tile `cells` with 2x2 blocks (8) and 3-lines (2)"""
        found = []

        def rec(rem, acc):
            if len(found) >= 2:
                return
            if not rem:
                m = {}
                for (pr, pc, kind, piece) in acc:
                    for cell in piece:
                        m[cell] = kind
                if m not in found:
                    found.append(m)
                return
            r, c = min(rem)
            cands = (
                ({(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}, 8),
                ({(r, c), (r, c + 1), (r, c + 2)}, 2),
                ({(r, c), (r + 1, c), (r + 2, c)}, 2),
            )
            for piece, kind in cands:
                if piece <= rem:
                    acc.append((r, c, kind, piece))
                    rec(rem - piece, acc)
                    acc.pop()
                    if len(found) >= 2:
                        return

        rec(set(cells), [])
        return found

    def unique_decomposition(gi, h, w):
        """decompose fgc cells of gi into objects; return None if impossible/ambiguous"""
        cells = {(r, c) for r in range(h) for c in range(w) if gi[r][c] != bgc}
        out = {}
        seen = set()
        for cell in sorted(cells):
            if cell in seen:
                continue
            comp = set()
            stack = [cell]
            seen.add(cell)
            while stack:
                r, c = stack.pop()
                comp.add((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            sols = colorings(comp)
            if len(sols) != 1:
                return None
            out.update(sols[0])
        return out

    hlo = min(8, max_h)
    wlo = min(8, max_w)
    for _attempt in range(40):
        boforb = set()
        reforb = set()
        h = unifint(diff_lb, diff_ub, (hlo, max_h))
        w = unifint(diff_lb, diff_ub, (wlo, max_w))
        gi = [[bgc] * w for _ in range(h)]
        go = [[bgc] * w for _ in range(h)]
        noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // 10)))
        inds = {(i, j) for i in range(h) for j in range(w)}
        for _k in range(noccs):
            obj, col = random.choice([(bo, 8), (random.choice([ro1, ro2]), 2)])
            oh = max(i for i, j in obj) + 1
            ow = max(j for i, j in obj) + 1
            forb = boforb if col == 8 else reforb
            cands = [
                (i, j) for (i, j) in inds
                if i <= h - oh and j <= w - ow
                and {(i + a, j + b) for a, b in obj} <= inds
                and (i, j) not in forb
            ]
            if not cands:
                break
            loc = random.choice(sorted(cands))
            li, lj = loc
            if col == 8:
                boforb.add((li - 2, lj))
                boforb.add((li + 2, lj))
                boforb.add((li, lj + 2))
                boforb.add((li, lj - 2))
            else:
                if obj == ro1:
                    reforb.add((li, lj + 3))
                    reforb.add((li, lj - 3))
                else:
                    reforb.add((li + 1, lj))
                    reforb.add((li - 1, lj))
            plcd = {(li + a, lj + b) for a, b in obj}
            for (r, c) in plcd:
                gi[r][c] = fgc
                go[r][c] = col
            inds -= plcd

        dec = unique_decomposition(gi, h, w)
        if dec is None:
            continue
        ok = all(go[r][c] == dec.get((r, c), bgc) for r in range(h) for c in range(w))
        if not ok:
            continue
        return {
            "input": tuple(tuple(row) for row in gi),
            "output": tuple(tuple(row) for row in go),
        }

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background = the canvas colour the generator paints before placing objects (majority)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    cells = {(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc}

    def tile(comp):
        """tile a component with 2x2 blocks (-> 8) and 1x3 / 3x1 lines (-> 2)"""
        result = []

        def rec(rem, acc):
            if result:
                return True
            if not rem:
                result.append(list(acc))
                return True
            r, c = min(rem)
            cands = (
                ((r, c, 1, 1, 8), {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}),
                ((r, c, 0, 2, 2), {(r, c), (r, c + 1), (r, c + 2)}),
                ((r, c, 2, 0, 2), {(r, c), (r + 1, c), (r + 2, c)}),
            )
            for spec, piece in cands:
                if piece <= rem:
                    acc.append(spec)
                    if rec(rem - piece, acc):
                        return True
                    acc.pop()
            return False

        rec(set(comp), [])
        return result[0] if result else []

    # walk fgc components; each component is one or more touching input objects
    seen = set()
    for cell in sorted(cells):
        if cell in seen:
            continue
        comp = set()
        stack = [cell]
        seen.add(cell)
        while stack:
            r, c = stack.pop()
            comp.add((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in cells and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        # recolour each input object as a whole: square -> 8, line -> 2
        for (r, c, dh, dw, col) in tile(comp):
            ops.append(col)
            sels.append([r, c, dh, dw])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task 150deff5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 150deff5"
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
                                f"for task 150deff5"
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
                    f"Failed to build a complete episode for task 150deff5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"150deff5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
