"""
ARC Task: cdecee7f (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(1, 9))
    h = random.randint(3, max(3, max_h))
    w = random.randint(3, max(3, max_w))
    numc = random.randint(1, min(9, w))
    locs = sorted(random.sample(range(w), numc))
    gi = [[bgc] * w for _ in range(h)]
    go = []
    for j in locs:
        iloc = random.randint(0, h - 1)
        col = random.choice(ccols)
        gi[iloc][j] = col
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = [go[:3], go[3:6][::-1], go[6:]]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background: marks are at most one per column, so bgc is the majority colour
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # Reading: every column is one unit; a column collapses to its single mark.
    # Marks, read left to right, are then laid out as a 3x3 snake:
    # row 0 left->right, row 1 right->left, row 2 left->right.
    marks = sorted([(r, c) for r in range(hi) for c in range(wi)
                    if int(I[r, c]) != bgc], key=lambda rc: rc[1])[:9]
    n = len(marks)
    slot = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
    final_slots = set(slot[:n])
    in_patch = lambda rc: rc[0] < 3 and rc[1] < 3

    G = I.copy()
    seen = {G.tobytes()}      # every grid state visited so far (no op may revisit one)
    ops, sels = [], []

    def find_path(cell, dest):
        """Shortest chain of unit Moves carrying the grabbed cell to dest such that
        every step leaves a grid state never seen before (no invisible / cyclic
        step).  Usually this is just the straight L-shaped slide."""
        from collections import deque
        snap = G.copy(); snap[cell] = 0
        colv = int(G[cell])
        def state(p):
            st = snap.copy(); st[p] = colv
            return st.tobytes()
        dirs = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        prev = {cell: None}
        used = {state(cell)}
        dq = deque([cell])
        while dq:
            p = dq.popleft()
            if p == dest:
                break
            for d in dirs:
                q = (p[0] + d[0], p[1] + d[1])
                if not (0 <= q[0] < hi and 0 <= q[1] < wi) or q in prev:
                    continue
                sq = state(q)
                if sq in seen or sq in used:
                    continue
                used.add(sq)
                prev[q] = (p, d)
                dq.append(q)
        path, p = [], dest
        while prev[p] is not None:
            p, d = prev[p]
            path.append(d)
        return path[::-1], snap, colv

    def slide(cell, dest):
        try:
            path, snap, colv = find_path(cell, dest)
        except KeyError:
            path = None
        if not path:   # no visibly-new route exists (never seen in practice): paint it over
            colv = int(G[cell])
            ops.append(colv); sels.append(sel_of([dest])); G[dest] = colv
            seen.add(G.tobytes())
            ops.append(bgc); sels.append(sel_of([cell])); G[cell] = bgc
            seen.add(G.tobytes())
            return
        cur = cell
        for j, (dr, dc) in enumerate(path):
            op = {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(dr, dc)]
            ops.append(op)
            sels.append(sel_of([cur]) if j == 0 else sel_of([]))   # grab once
            cur = (cur[0] + dr, cur[1] + dc)
            st = snap.copy(); st[cur] = colv
            seen.add(st.tobytes())
        G[:, :] = st

    def repair(cells):
        cells = [rc for rc in cells if in_patch(rc) and rc not in final_slots
                 and int(G[rc]) != bgc]
        if cells:
            ops.append(bgc); sels.append(sel_of(cells))
            for rc in cells:
                G[rc] = bgc
            seen.add(G.tobytes())

    zero_marks = [i for i, (r, c) in enumerate(marks) if int(I[r, c]) == 0]
    # Each mark (the one cell of its column), read left to right, slides straight
    # to its snake slot.  Marks 0..2 go first (they are the only ones that can sit
    # in the 3x3 corner); row 1 is filled right-to-left so slides don't cross it.
    order = [k for k in (0, 1, 2, 5, 4, 3, 6, 7, 8) if k < n and k not in zero_marks]
    for k in order:
        r, c = marks[k]
        dr, dc = slot[k]
        if (r, c) == (dr, dc):
            continue
        slide((r, c), (dr, dc))
        repair([(r, c)])
    # 0-coloured marks can't be grabbed (0 = empty to Move): paint them into
    # their slot and clear their source.
    if zero_marks:
        repair([marks[i] for i in zero_marks])
        cells = [slot[i] for i in zero_marks if int(G[slot[i]]) != 0]
        if cells:
            ops.append(0); sels.append(sel_of(cells))
            for rc in cells:
                G[rc] = 0

    if not (hi == 3 and wi == 3):
        ops.append(33); sels.append([0, 0, 2, 2])   # crop to the full 3x3 answer patch
    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task cdecee7f"
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
                                f"for task cdecee7f"
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
                    f"Failed to build a complete episode for task cdecee7f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"cdecee7f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
