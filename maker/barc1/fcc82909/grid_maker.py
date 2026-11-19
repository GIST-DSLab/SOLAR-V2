"""
ARC Task: fcc82909 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 3 is hardcoded as the tail colour, so the background never uses it.
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]

    h = unifint(diff_lb, diff_ub, (7, max(7, max_h)))
    w = unifint(diff_lb, diff_ub, (7, max(7, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, w // 3)))

    opts = list(range(w))
    tr = 0
    maxtr = 4 * nobjs
    succ = 0
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    while succ < nobjs and tr < maxtr:
        tr += 1
        sopts = [j for j in opts if j + 1 in opts]
        if not sopts:
            break
        numc = unifint(diff_lb, diff_ub, (1, 4))
        jstart = random.choice(sopts)
        opts = [j for j in opts if j not in (jstart, jstart + 1)]
        options = list(range(0, h - 2 - numc + 1))
        if not options:
            break
        iloc = random.choice(options)
        ccols = random.sample(remcols, numc)
        bd = [(iloc + a, jstart + b) for a in range(2) for b in range(2)]
        random.shuffle(bd)
        obj = [(c, ij) for c, ij in zip(ccols, bd[:numc])] + \
              [(random.choice(ccols), ij) for ij in bd[numc:]]
        ok = True
        for _, (r, c) in obj:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and gi[nr][nc] != bgc:
                    ok = False
        if not ok:
            continue
        for col, (r, c) in obj:
            gi[r][c] = col
            go[r][c] = col
        for k in range(numc):
            rr = iloc + k + 2
            if rr < h:
                go[rr][jstart] = 3
                go[rr][jstart + 1] = 3
        succ += 1

    # Recompute the output with the declarative rule, so overlap edge cases match the verifier.
    G = np.array(gi)
    out = G.copy()
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if G[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                cr, cc = q.popleft(); cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and G[nr, nc] != bgc:
                        seen[nr, nc] = True; q.append((nr, nc))
            r1 = max(p[0] for p in cells)
            c0 = min(p[1] for p in cells)
            nc_ = len({int(G[p]) for p in cells})
            for rr in range(r1 + 1, min(h, r1 + 1 + nc_)):
                for cc in (c0, c0 + 1):
                    if cc < w:
                        out[rr, cc] = 3
    return {"input": G.tolist(), "output": out.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # Background: the canvas colour (objects are tiny 2x2 squares), so majority is reliable.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Each 4-connected multicolour non-bg component is one 2x2 square.
    seen = np.zeros((hi, wi), dtype=bool)
    objs = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            q = deque([(r, c)]); seen[r, c] = True; cells = []
            while q:
                cr, cc = q.popleft(); cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True; q.append((nr, nc))
            objs.append(cells)

    cur = I.copy()
    # Under each square, draw a green rectangle 2 wide and L tall, where L = number of
    # distinct colours in the square. The rectangle is clipped at the bottom edge.
    for cells in objs:
        r1 = max(p[0] for p in cells)
        c0 = min(p[1] for p in cells)
        numc = len({int(I[p]) for p in cells})
        rect = [(rr, cc) for rr in range(r1 + 1, min(hi, r1 + 1 + numc))
                for cc in (c0, c0 + 1) if cc < wi]
        if not rect:
            continue
        if all(cur[p] == 3 for p in rect):
            continue  # already green (overlapping tail) -> op would be a no-op
        ops.append(3); sels.append(sel_of(rect))
        for p in rect:
            cur[p] = 3

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
                        f"num_examples+1 ({num_examples + 1}) for task fcc82909"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task fcc82909"
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
                                f"for task fcc82909"
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
                    f"Failed to build a complete episode for task fcc82909 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"fcc82909-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
