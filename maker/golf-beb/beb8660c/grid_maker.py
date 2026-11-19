"""
ARC Task: beb8660c (RE-ARC) — LLM-generated grid_maker
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

import random
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice([c for c in range(10) if c != 8])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, rot=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c != 8])
    if rot is None:
        rot = random.choice(ROTS)
    cols = [c for c in range(10) if c != 8]
    mh, mw = (max_w, max_h) if rot in (1, 3) else (max_h, max_w)
    w = unifint(diff_lb, diff_ub, (3, min(30, mw)))
    h = unifint(diff_lb, diff_ub, (w, max(w, min(30, mh))))
    if h > mh or w > mw:
        raise ValueError("too big")
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    k = unifint(diff_lb, diff_ub, (1, min(8, w - 1)))
    co = random.sample(remcols, k)
    wds = sorted(random.sample(range(1, w), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        jj = h - k - 1 + j
        for cc in range(l):
            gi[jj][cc] = c
    gi[h - 1] = [8] * w
    go = [row[::-1] for row in gi]
    rows = [list(r) for r in gi[:-1]]
    random.shuffle(rows)
    gif = []
    for r in rows:
        nbc = r.count(bgc)
        ofs = random.randint(0, nbc)
        gif.append(r[-ofs:] + r[:-ofs] if ofs else list(r))
    gif.append(list(go[-1]))
    gi = [row[::-1] for row in gif]
    gi = np.rot90(np.array(gi), -rot)
    go = np.rot90(np.array(go), -rot)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # the 8 wall: a full edge row/column
    if np.all(I[0, :] == 8):
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8):
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    else:
        edge = 'right'
    # background: most common non-8 colour (each bar is shorter than the free span)
    cnt = Counter(int(v) for v in I.flatten() if v != 8)
    bgc = cnt.most_common(1)[0][0]

    # each bar = all cells of one non-bg, non-8 colour (a straight segment)
    bars = {}
    for c in sorted(cnt):
        if c == bgc:
            continue
        bars[c] = [(int(r), int(cc)) for r, cc in zip(*np.where(I == c))]
    order = sorted(bars, key=lambda c: (-len(bars[c]), c))

    def target(i, L):
        if edge == 'top':
            return [(1 + i, c) for c in range(L)]
        if edge == 'bottom':
            return [(hi - 2 - i, c) for c in range(wi - L, wi)]
        if edge == 'left':
            return [(r, 1 + i) for r in range(hi - L, hi)]
        return [(r, wi - 2 - i) for r in range(L)]

    tgt = {c: target(i, len(bars[c])) for i, c in enumerate(order)}
    pos = {c: list(bars[c]) for c in order}
    grid = I.copy()
    placed = set()
    horizontal = edge in ('top', 'bottom')

    def free(cells, me):
        return all(grid[r, c] == bgc or (r, c) in set(pos[me]) for r, c in cells)

    def relocate(c, dest):
        src = pos[c]
        dr = dest[0][0] - src[0][0]
        dc = dest[0][1] - src[0][1]
        if c == 0:
            # colour 0 cannot be grabbed by ARCLE's object ops: erase and redraw
            ops.append(bgc); sels.append(sel_of(src))
            for r, cc in src:
                grid[r, cc] = bgc
            ops.append(0); sels.append(sel_of(dest))
            for r, cc in dest:
                grid[r, cc] = 0
        else:
            steps = []
            # move across rows/cols first, then along the bar's own axis
            if horizontal:
                steps += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
                steps += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
            else:
                steps += [(23 if dc < 0 else 22, (0, -1 if dc < 0 else 1))] * abs(dc)
                steps += [(20 if dr < 0 else 21, (-1 if dr < 0 else 1, 0))] * abs(dr)
            for i, (op, _) in enumerate(steps):
                ops.append(op)
                sels.append(sel_of(src) if i == 0 else sel_of([]))
            hole = sorted(set(src) - set(dest))
            for r, cc in src:
                grid[r, cc] = bgc
            for r, cc in dest:
                grid[r, cc] = c
            if bgc != 0 and hole:
                ops.append(bgc); sels.append(sel_of(hole))
        pos[c] = list(dest)

    def spots(c):
        # every straight placement of bar c parallel to its own axis
        L = len(bars[c])
        out = []
        if horizontal:
            for r in range(hi):
                if np.all(I[r, :] == 8):
                    continue
                for s in range(wi - L + 1):
                    out.append([(r, s + t) for t in range(L)])
        else:
            for cc in range(wi):
                if np.all(I[:, cc] == 8):
                    continue
                for s in range(hi - L + 1):
                    out.append([(s + t, cc) for t in range(L)])
        return out

    guard = 0
    while len(placed) < len(order):
        guard += 1
        if guard > 200:
            raise ValueError("could not schedule moves")
        progressed = False
        for c in order:  # longest first, nearest to the wall first
            if c in placed:
                continue
            if free(tgt[c], c):
                if sorted(pos[c]) != sorted(tgt[c]):
                    relocate(c, tgt[c])
                placed.add(c)
                progressed = True
                break
        if progressed:
            continue
        # every remaining bar is blocked: step a blocker aside
        done = False
        for c in order:
            if c in placed:
                continue
            tset = set(tgt[c])
            blockers = [b for b in order if b not in placed and b != c and tset & set(pos[b])]
            if len(blockers) != 1:
                continue
            b = blockers[0]
            other_t = set()
            for o in order:
                if o not in placed and o != b:
                    other_t |= set(tgt[o])
            cands = []
            for sp in spots(b):
                sps = set(sp)
                if sps & tset or not free(sp, b):
                    continue
                cost = len(sps & other_t) * 1000 + abs(sp[0][0] - pos[b][0][0]) + abs(sp[0][1] - pos[b][0][1])
                cands.append((cost, sp))
            if cands:
                cands.sort(key=lambda x: x[0])
                relocate(b, cands[0][1])
                done = True
                break
        if not done:
            raise ValueError("deadlock")

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task beb8660c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task beb8660c"
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
                                f"for task beb8660c"
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
                    f"Failed to build a complete episode for task beb8660c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"beb8660c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
