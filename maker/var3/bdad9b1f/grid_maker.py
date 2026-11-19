"""
ARC Task: bdad9b1f (RE-ARC) — LLM-generated grid_maker
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
    # bgc is the only sampled colour role the rule leans on (line colours are
    # carried through unchanged, and the crossing marker 4 is hardcoded).
    # The generator's rot choice is a discrete structural variant (it decides which
    # edge the stubs grow from), so it is pre-planned per instance.
    bgc = random.choice([c for c in range(10) if c != 4])
    n_ex = num_examples if num_examples else 3
    variants = [{"rot": r} for r in ("identity", "rot90", "rot180", "rot270")]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    def unif(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(lo, b))
        hi = max(lo, min(hi, b))
        return random.randint(lo, hi)

    def rot90(g):       # CW
        return [list(r) for r in zip(*g[::-1])]

    def rot180(g):
        return [list(r[::-1]) for r in g[::-1]]

    def rot270(g):      # CCW
        return [list(r) for r in zip(*g)][::-1]

    if rot is None:
        rot = random.choice(("identity", "rot90", "rot180", "rot270"))

    cols = [c for c in range(10) if c != 4]

    # a 90/270 rotation swaps the dims, so cap h/w with that in mind
    if rot in ("rot90", "rot270"):
        hcap, wcap = min(30, max_w), min(30, max_h)
    else:
        hcap, wcap = min(30, max_h), min(30, max_w)
    hcap = max(4, hcap)
    wcap = max(4, wcap)

    h = unif(diff_lb, diff_ub, (4, hcap))
    w = unif(diff_lb, diff_ub, (4, wcap))

    numh = max(1, min(unif(diff_lb, diff_ub, (1, h // 2 - 1)), h - 3))
    numw = max(1, min(unif(diff_lb, diff_ub, (1, w // 2 - 1)), w - 3))

    hlocs = sorted(random.sample(list(range(2, h - 1)), numh))
    wlocs = sorted(random.sample(list(range(2, w - 1)), numw))

    remcols = [c for c in cols if c != bgc]
    numcols = min(unif(diff_lb, diff_ub, (2, 8)), len(remcols))
    ccols = random.sample(remcols, numcols)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    fc = -1
    for ii in hlocs:
        col = random.choice([c for c in ccols if c != fc])
        fc = col
        objw = random.randint(2, ii)
        for jj in range(min(objw, w)):
            gi[ii][jj] = col
        for jj in range(w):
            go[ii][jj] = col

    fc = -1
    for jj in wlocs:
        col = random.choice([c for c in ccols if c != fc])
        fc = col
        objh = random.randint(2, jj)
        for ii in range(min(objh, h)):
            gi[ii][jj] = col
        for ii in range(h):
            go[ii][jj] = col

    for ii in hlocs:
        for jj in wlocs:
            go[ii][jj] = 4

    if rot == "rot90":
        gi, go = rot90(gi), rot90(go)
    elif rot == "rot180":
        gi, go = rot180(gi), rot180(go)
    elif rot == "rot270":
        gi, go = rot270(gi), rot270(go)

    return {
        "input": tuple(tuple(int(v) for v in row) for row in gi),
        "output": tuple(tuple(int(v) for v in row) for row in go),
    }


def derive_operations(I, O, examples=None):
    """Every stub is drawn out into a full line; every crossing is then marked.

    The set of actions is forced by the rule, but the route through them is not:
    which family of stubs to finish first, which end of the grid to sweep from,
    and whether a crossing is marked as its second line lands or all of them are
    marked together in one final pass.  Each of those is read off the input (how
    many stubs of each kind, where the longest one sits) — never off O.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def background(G):
        # Stubs only ever sit in rows 2..h-2 / cols 2..w-2, and the generator's
        # rotation keeps corners at corners, so all four corners are background.
        gh, gw = G.shape
        cnt = Counter([int(G[0, 0]), int(G[0, gw - 1]),
                       int(G[gh - 1, 0]), int(G[gh - 1, gw - 1])])
        val, n = cnt.most_common(1)[0]
        if n >= 2:
            return val
        return int(Counter(G.reshape(-1).tolist()).most_common(1)[0][0])

    def stubs(G):
        """row -> (colour, length) for horizontal stubs, col -> (…) for vertical."""
        gh, gw = G.shape
        bg = background(G)
        rows, cols = {}, {}
        for r in range(gh):
            c = 0
            while c < gw:
                v = int(G[r, c])
                if v == bg:
                    c += 1
                    continue
                c2 = c
                while c2 + 1 < gw and int(G[r, c2 + 1]) == v:
                    c2 += 1
                if c2 - c + 1 >= 2:
                    rows[r] = (v, c2 - c + 1)
                c = c2 + 1
        for c in range(gw):
            r = 0
            while r < gh:
                v = int(G[r, c])
                if v == bg:
                    r += 1
                    continue
                r2 = r
                while r2 + 1 < gh and int(G[r2 + 1, c]) == v:
                    r2 += 1
                if r2 - r + 1 >= 2:
                    cols[c] = (v, r2 - r + 1)
                r = r2 + 1
        return rows, cols

    rowstubs, colstubs = stubs(I)

    # ── the colour a crossing is marked with is a convention I alone does not fix:
    #    read it off the demonstrations, where two extended lines meet.
    marker = 4
    if examples:
        seen = []
        for pair in examples:
            try:
                if isinstance(pair, dict):
                    eI, eO = pair["input"], pair["output"]
                else:
                    eI, eO = pair[0], pair[1]
                eI = np.asarray(eI, dtype=int)
                eO = np.asarray(eO, dtype=int)
            except Exception:
                continue
            if eI.shape != eO.shape:
                continue
            er, ec = stubs(eI)
            for r in er:
                for c in ec:
                    seen.append(int(eO[r, c]))
        if seen:
            marker = int(Counter(seen).most_common(1)[0][0])

    # ── route choice 1: finish the bigger family of stubs first; if both families
    #    are the same size, the one already showing more colour on the grid.
    n_r, n_c = len(rowstubs), len(colstubs)
    bulk_r = sum(v[1] for v in rowstubs.values())
    bulk_c = sum(v[1] for v in colstubs.values())
    if n_r != n_c:
        rows_first = n_r > n_c
    elif bulk_r != bulk_c:
        rows_first = bulk_r > bulk_c
    else:
        rows_first = True

    # ── route choice 2: sweep each family starting from the edge that family's
    #    longest stub lies nearest to.
    def sweep(table, extent):
        if not table:
            return []
        anchor = max(table, key=lambda i: (table[i][1], i))
        far_side = (2 * anchor > extent - 1)
        return sorted(table, reverse=far_side)

    rows_seq = sweep(rowstubs, h)
    cols_seq = sweep(colstubs, w)
    first_seq, second_seq = (rows_seq, cols_seq) if rows_first else (cols_seq, rows_seq)

    # ── route choice 3: if each line of the second family crosses several of the
    #    first, mark its crossings the moment it is drawn; if it only ever crosses
    #    one, leave every mark to a single pass at the end.
    per_line_marks = len(first_seq) >= 2 and len(second_seq) >= 1

    G = I.copy()
    ops, sels = [], []

    def draw(idx, is_row):
        colr = (rowstubs if is_row else colstubs)[idx][0]
        if is_row:
            cells = [(idx, c) for c in range(w) if int(G[idx, c]) != colr]
        else:
            cells = [(r, idx) for r in range(h) if int(G[r, idx]) != colr]
        if not cells:            # this stub already spans the grid — nothing to draw
            return
        ops.append(int(colr))            # Color<line colour> on the line's own cells
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = colr

    def mark(cells):
        cells = [(r, c) for (r, c) in cells if int(G[r, c]) != marker]
        if not cells:
            return
        ops.append(int(marker))          # Color<marker> on the crossings
        sels.append(sel_of(cells))
        for (r, c) in cells:
            G[r, c] = marker

    for idx in first_seq:
        draw(idx, rows_first)

    for idx in second_seq:
        draw(idx, not rows_first)
        if per_line_marks:
            if rows_first:                       # idx is a column, first_seq are rows
                mark([(r, idx) for r in first_seq])
            else:                                # idx is a row, first_seq are columns
                mark([(idx, c) for c in first_seq])

    if not per_line_marks:
        mark([(r, c) for r in rows_seq for c in cols_seq])

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])   # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task bdad9b1f"
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
                                f"for task bdad9b1f"
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
                    f"Failed to build a complete episode for task bdad9b1f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"bdad9b1f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
