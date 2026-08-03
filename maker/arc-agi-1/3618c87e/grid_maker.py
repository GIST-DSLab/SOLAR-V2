"""
ARC Task: 3618c87e (RE-ARC) — LLM-generated grid_maker
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

ROT_VARIANTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    # dotc must be non-zero: ARCLE Move grabs only NONZERO cells of the selection,
    # and the moving dot is a single cell.
    dotc = random.choice([c for c in cols if c != 0])
    rest = [c for c in cols if c != dotc]
    bgc, linc = random.sample(rest, 2)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [{"rotf": r} for r in ROT_VARIANTS]
        examples += [{"rotf": random.choice(ROT_VARIANTS)} for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"rotf": r} for r in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "dotc": dotc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, dotc, rotf=None) -> dict:
    def _rot90(g):      # clockwise
        return tuple(tuple(row) for row in zip(*g[::-1]))

    def _rot180(g):
        return tuple(tuple(r[::-1]) for r in g[::-1])

    def _rot270(g):     # counter-clockwise
        return tuple(tuple(row) for row in list(zip(*g))[::-1])

    def _ident(g):
        return tuple(tuple(r) for r in g)

    if rotf is None:
        rotf = random.choice(ROT_VARIANTS)

    hb, wb = max_h, max_w
    if rotf in ("rot90", "rot270"):
        m = min(max_h, max_w)
        hb = wb = m
    hb = max(4, hb)
    wb = max(4, wb)

    h = unifint(diff_lb, diff_ub, (4, hb))
    w = unifint(diff_lb, diff_ub, (4, wb))

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    # base line: whole row 0
    for j in range(w):
        gi[0][j] = linc
        go[0][j] = linc

    nlocs = unifint(diff_lb, diff_ub, (1, max(1, w // 2)))
    locs = []
    opts = list(range(w))
    for _ in range(nlocs):
        if not opts:
            break
        ch = random.choice(opts)
        locs.append(ch)
        opts = [o for o in opts if o not in (ch - 1, ch, ch + 1)]

    for j in locs:
        hh = random.randint(1, h - 3)
        for r in range(0, hh + 1):
            gi[r][j] = linc
            go[r][j] = linc
        gi[hh + 1][j] = dotc   # dot sits just past the tip of the stem
        go[0][j] = dotc        # dot ends up at the base line

    rf = {"identity": _ident, "rot90": _rot90, "rot180": _rot180, "rot270": _rot270}[rotf]
    return {'input': rf(gi), 'output': rf(go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    colors = sorted(set(int(v) for v in I.flatten().tolist()))
    cells_of = {c: [(r, cc) for r in range(h) for cc in range(w) if int(I[r, cc]) == c]
                for c in colors}

    # --- dot colour: the only colour whose every cell is an isolated singleton ---
    def all_isolated(col):
        s = set(cells_of[col])
        if not s:
            return False
        for (r, cc) in s:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (r + dr, cc + dc) in s:
                    return False
        return True

    cands = [c for c in colors if all_isolated(c)]
    if not cands:
        cands = colors
    dotc = min(cands, key=lambda c: len(cells_of[c]))
    dots = sorted(cells_of[dotc])

    # --- base edge: a fully constant edge whose adjacent inner line carries that
    #     colour at EXACTLY the dots' perpendicular coordinates (the stems) ---
    edge_specs = [
        ('U', [(0, j) for j in range(w)],     [(1, j) for j in range(w)]),
        ('D', [(h - 1, j) for j in range(w)], [(h - 2, j) for j in range(w)]),
        ('L', [(i, 0) for i in range(h)],     [(i, 1) for i in range(h)]),
        ('R', [(i, w - 1) for i in range(h)], [(i, w - 2) for i in range(h)]),
    ]
    base_name, linc = None, None
    for name, edge, inner in edge_specs:
        vals = set(int(I[r, c]) for r, c in edge)
        if len(vals) != 1:
            continue
        A = vals.pop()
        if A == dotc:
            continue
        if name in ('U', 'D'):
            innerpos = set(j for (r, j) in inner if int(I[r, j]) == A)
            dotpos = set(c for (r, c) in dots)
        else:
            innerpos = set(i for (i, c) in inner if int(I[i, c]) == A)
            dotpos = set(r for (r, c) in dots)
        if innerpos and innerpos == dotpos:
            base_name, linc = name, A
            break
    if base_name is None:
        # fallback: side of the unique neighbour colour of the first dot
        r0, c0 = dots[0]
        nb = []
        for name, (dr, dc) in (('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))):
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < h and 0 <= cc < w:
                nb.append((name, int(I[rr, cc])))
        cnt = Counter(v for _, v in nb)
        base_name, linc = next((n, v) for n, v in nb if cnt[v] == 1)

    rest = [c for c in colors if c != dotc and c != linc]
    bgc = max(rest, key=lambda c: len(cells_of[c])) if rest else 0

    if base_name == 'U':
        mop, (dr, dc) = 20, (-1, 0)
        nsteps = lambda r, c: r
    elif base_name == 'D':
        mop, (dr, dc) = 21, (1, 0)
        nsteps = lambda r, c: h - 1 - r
    elif base_name == 'L':
        mop, (dr, dc) = 23, (0, -1)
        nsteps = lambda r, c: c
    else:
        mop, (dr, dc) = 22, (0, 1)
        nsteps = lambda r, c: w - 1 - c

    # --- slide each dot along its stem until it reaches the base line ---
    for (r, c) in dots:
        n = nsteps(r, c)
        if n <= 0:
            continue
        cur = (r, c)
        ops.append(mop); sels.append(sel_of([cur]))          # grab the dot
        cur = (cur[0] + dr, cur[1] + dc)
        for _ in range(n - 1):
            ops.append(mop); sels.append(sel_of([]))         # keep the same object grabbed
            cur = (cur[0] + dr, cur[1] + dc)
        # ARCLE zeroed only the dot's ORIGINAL footprint (the path is restored)
        if bgc != 0 and cur != (r, c):
            ops.append(int(bgc)); sels.append(sel_of([(r, c)]))

    ops.append(34); sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3618c87e"
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
                                f"for task 3618c87e"
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
                    f"Failed to build a complete episode for task 3618c87e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3618c87e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
