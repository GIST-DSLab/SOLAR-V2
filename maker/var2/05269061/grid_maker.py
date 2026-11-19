"""
ARC Task: 05269061 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """
    The rule of 05269061 depends only on the *pattern* of the seed lines, never on
    which colour a line carries, and the canvas colour is hardcoded 0 in the
    generator -- so no colour role has to be pinned.  What DOES have to be planned
    is the structural family: full rows, anti-diagonals, or (after the generator's
    vmirror) main diagonals.  Every family that can show up in the test instance is
    covered by the examples.
    """
    variants = [{"mode": "rows"}, {"mode": "anti"}, {"mode": "main"}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, mode=None, **kwargs) -> dict:
    if mode is None:
        mode = random.choice(["rows", "anti", "main"])

    def _uni(bounds):
        a, b = bounds
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    dmax = min(30, int(max_h), int(max_w))
    if dmax < 3:
        dmax = 3
    colopts = list(range(1, 10))

    gi = go = None
    for _attempt in range(64):
        d = _uni((3, dmax))
        gi = [[0] * d for _ in range(d)]
        go = [[0] * d for _ in range(d)]
        if mode == "rows":
            K = d
            # at least one row must stay empty in the input, else input == output
            num = _uni((2, min(d - 1, 9)))

            def cells(k):
                return [(k, c) for c in range(d)]
        else:
            K = 2 * d - 1
            num = _uni((2, min(2 * d - 2, 9)))

            def cells(k):
                return [(k - t, t) for t in range(d) if 0 <= k - t < d]

        cols = [random.choice(colopts) for _ in range(num)]
        keeps = [random.choice(list(range(j, K, num))) for j in range(num)]
        for k in range(K):
            col = cols[k % num]
            cs = cells(k)
            for (r, c) in cs:
                go[r][c] = col
            if keeps[k % num] == k:
                for (r, c) in cs:
                    gi[r][c] = col
        if mode == "main":
            gi = [list(reversed(row)) for row in gi]
            go = [list(reversed(row)) for row in go]
        if gi != go:
            break

    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Rule (read off the input alone):
      * the input shows a handful of monochrome seed LINES -- either full rows, or
        full anti-diagonals (r+c const), or full main diagonals (r-c const);
      * the number of seed lines IS the period p of the family;
      * a seed on line k0 colours every line k with k % p == k0 % p.

    The ROUTE taken to do that is decided by the picture itself, so the same input
    always gives the same route, and different inputs give different routes:

      * the colour of the seed line nearest the top-left corner ("anchor colour"):
          odd  -> the tidy way: one selection per colour (all of that colour's
                  lines at once), on top of a base layer when one colour marks
                  two or more of the classes;
          even -> the builder's way: if the period block tiles the picture three
                  or more times, lay the first block and copy it across;
                  otherwise sweep line by line outward from each seed.
      * the grid size plus the number of seeds choose which seed to start from
        (nearest corner / farthest corner / lowest colour first).
      * each seed's lines are swept toward the side that has more lines to fill.

    Nothing here looks at O except to size the Submit selection.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    d, w = I.shape
    ho, wo = O.shape
    G = I.copy()
    ops, sels = [], []

    def paint(cells, col):
        # skip cells that already hold this colour at this moment
        tgt = [(int(r), int(c)) for (r, c) in cells if G[r, c] != col]
        if not tgt:
            return
        ops.append(int(col))
        sels.append(sel_of(tgt))
        for (r, c) in tgt:
            G[r, c] = col

    def row_cells(k):
        return [(k, c) for c in range(w)]

    def anti_cells(k):
        return [(r, k - r) for r in range(d) if 0 <= k - r < w]

    def main_cells(k):
        return [(r, r - k + d - 1) for r in range(d) if 0 <= r - k + d - 1 < w]

    # ---- which family of lines is this? -------------------------------------
    nz = [(r, c) for r in range(d) for c in range(w) if I[r, c] != 0]
    rws = sorted({r for r, _ in nz})
    horiz = bool(nz) and len(nz) == len(rws) * w and all(
        all(I[r, c] == I[r, 0] and I[r, c] != 0 for c in range(w)) for r in rws
    )
    if horiz:
        fam, cells_of, K = 'row', row_cells, d
        idx_of = lambda r, c: r
        seeds = sorted((r, int(I[r, 0])) for r in rws)
    else:
        def try_orient(cells_fn, key_fn):
            groups = {}
            for (r, c) in nz:
                groups.setdefault(key_fn(r, c), []).append((r, c))
            for k, g in groups.items():
                if set(g) != set(cells_fn(k)):
                    return None
                if len({int(I[r, c]) for r, c in g}) != 1:
                    return None
            return groups

        K = d + w - 1
        got = try_orient(anti_cells, lambda r, c: r + c)
        if got is not None:
            fam, cells_of = 'anti', anti_cells
            idx_of = lambda r, c: r + c
        else:
            got = try_orient(main_cells, lambda r, c: r - c + d - 1)
            fam, cells_of = 'main', main_cells
            idx_of = lambda r, c: r - c + d - 1
        seeds = sorted((k, int(I[g[0][0], g[0][1]])) for k, g in got.items())

    if not seeds:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels

    p = len(seeds)
    col_of, seed_k = {}, {}
    for k0, col in seeds:
        col_of.setdefault(k0 % p, col)
        seed_k.setdefault(k0 % p, k0)

    # ---- the choices, all read off the input --------------------------------
    anchor_col = seeds[0][1]                 # colour of the seed nearest top-left
    batch = (anchor_col % 2 == 1)
    counts = Counter(col for _, col in seeds)
    base_col, dup = None, 1
    for k0, col in seeds:                    # seeds are in position order
        if counts[col] > dup:
            dup, base_col = counts[col], col
    reps = (d if fam == 'row' else w) // p   # how often the period block tiles

    if batch:
        route = 'base' if base_col is not None else 'plain'
    else:
        if reps >= 3:
            route = 'copy'
        elif base_col is not None:
            route = 'base'
        else:
            route = 'plain'

    order_key = (d + p) % 3
    seeds_by_k = sorted(seeds)
    if order_key == 0:
        seq = seeds_by_k                                        # nearest corner first
    elif order_key == 1:
        seq = list(reversed(seeds_by_k))                        # farthest first
    else:
        seq = sorted(seeds_by_k, key=lambda s: (s[1], s[0]))    # lowest colour first
    order, seen = [], set()
    for k0, _c in seq:
        res = k0 % p
        if res not in seen:
            seen.add(res)
            order.append(res)

    def sweep(ks, k0):
        """the seed's own line first, then outward, bigger side first"""
        mid = [k for k in ks if k == k0]
        lo = sorted([k for k in ks if k < k0], reverse=True)
        hi = sorted([k for k in ks if k > k0])
        return mid + (hi + lo if len(hi) >= len(lo) else lo + hi)

    # ---- route ---------------------------------------------------------------
    if route != 'copy':
        if route == 'base':
            # lay the dominant colour over the picture, then draw the rest on top
            paint([(r, c) for r in range(d) for c in range(w)], base_col)
        for res in order:
            col = col_of[res]
            if route == 'base' and col == base_col:
                continue                      # those classes are the base already
            ks = [k for k in range(K) if k % p == res]
            if route == 'plain':
                ks = [k for k in ks if k != seed_k[res]]   # the seed line is already right
            if not ks:
                continue
            if batch:
                cells = [cell for k in ks for cell in cells_of(k)]
                paint(cells, col)
            else:
                for k in sweep(ks, seed_k[res]):
                    paint(cells_of(k), col)
    else:
        if fam == 'row':
            nfull = d // p
            strip = [(r, c) for r in range(p) for c in range(w)]
            box = [0, 0, p - 1, w - 1]        # exactly this full rectangle
            tail_n = d - nfull * p
        else:
            nfull = w // p
            strip = [(r, c) for r in range(d) for c in range(p)]
            box = [0, 0, d - 1, p - 1]        # exactly this full rectangle
            tail_n = w - nfull * p

        # 1. lay the first period block, one colour class at a time
        for res in order:
            cells = [(r, c) for (r, c) in strip if idx_of(r, c) % p == res]
            paint(cells, col_of[res])

        # cells of the block that hold colour 0: Copy/Paste cannot carry them
        zeros = [(r, c) for (r, c) in strip if G[r, c] == 0]

        # 2. stamp the block along the picture
        copied = False
        for t in range(1, nfull):
            dr = p * t if fam == 'row' else 0
            dc = 0 if fam == 'row' else p * t
            if any(G[r, c] != 0 and G[r + dr, c + dc] != G[r, c] for (r, c) in strip):
                if not copied:
                    ops.append(29)
                    sels.append(box)
                    copied = True
                ops.append(30)
                sels.append([dr, dc, 0, 0])
                for (r, c) in strip:
                    if G[r, c] != 0:
                        G[r + dr, c + dc] = G[r, c]
            if zeros:
                paint([(r + dr, c + dc) for (r, c) in zeros], 0)

        # 3. the part that is narrower than a whole block: copy just what fits
        if tail_n:
            if fam == 'row':
                tbox = [0, 0, tail_n - 1, w - 1]
                dr, dc = nfull * p, 0
                tcells = [(r, c) for r in range(tail_n) for c in range(w)]
            else:
                tbox = [0, 0, d - 1, tail_n - 1]
                dr, dc = 0, nfull * p
                tcells = [(r, c) for r in range(d) for c in range(tail_n)]
            if any(G[r, c] != 0 and G[r + dr, c + dc] != G[r, c] for (r, c) in tcells):
                ops.append(29)
                sels.append(tbox)
                ops.append(30)
                sels.append([dr, dc, 0, 0])
                for (r, c) in tcells:
                    if G[r, c] != 0:
                        G[r + dr, c + dc] = G[r, c]
            tz = [(r + dr, c + dc) for (r, c) in tcells if G[r, c] == 0]
            if tz:
                paint(tz, 0)

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
                        f"num_examples+1 ({num_examples + 1}) for task 05269061"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 05269061"
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
                                f"for task 05269061"
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
                    f"Failed to build a complete episode for task 05269061 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"05269061-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
