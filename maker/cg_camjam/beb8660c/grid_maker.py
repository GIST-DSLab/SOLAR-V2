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

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
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
    if k >= h - 1:
        raise ValueError("no spare row")
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

    # the 8 wall: a full edge row/column of colour 8
    if np.all(I[0, :] == 8):
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8):
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    else:
        edge = 'right'
    cnt = Counter(int(v) for v in I.flatten() if v != 8)
    bgc = cnt.most_common(1)[0][0]

    # canonical frame: wall at the bottom, bars horizontal, gravity to the right
    if edge in ('top', 'bottom'):
        H, W = hi, wi
    else:
        H, W = wi, hi

    def to_act(r, c):
        if edge == 'bottom':
            return (r, c)
        if edge == 'top':
            return (hi - 1 - r, wi - 1 - c)
        if edge == 'left':
            return (c, H - 1 - r)
        return (W - 1 - c, r)

    DIR = {'bottom': {'R': 22, 'D': 21, 'U': 20, 'L': 23},
           'top':    {'R': 23, 'D': 20, 'U': 21, 'L': 22},
           'left':   {'R': 21, 'D': 23, 'U': 22, 'L': 20},
           'right':  {'R': 20, 'D': 22, 'U': 23, 'L': 21}}[edge]
    DELTA = {'R': (0, 1), 'L': (0, -1), 'U': (-1, 0), 'D': (1, 0)}

    to_can = {to_act(r, c): (r, c) for r in range(H) for c in range(W)}
    bars = {}
    for col in cnt:
        if col == bgc:
            continue
        cells = [to_can[(int(r), int(c))] for r, c in zip(*np.where(I == col))]
        bars[col] = (cells[0][0], min(c for _, c in cells), len(cells))  # row, start, length

    # canonical-frame simulation of the working grid (from I only)
    G = np.full((H, W), bgc, dtype=int)
    G[H - 1, :] = 8
    for col, (row, start, L) in bars.items():
        G[row, start:start + L] = col
    vacated = set()   # footprint cells a bar has left (ARCLE leaves them at 0)

    def act(cells):
        return sel_of([to_act(r, c) for r, c in cells])

    def cells_of(row, start, L):
        return [(row, start + t) for t in range(L)]

    def slide(col, d, n):
        row, start, L = bars[col]
        src = cells_of(row, start, L)
        dr, dc = DELTA[d]
        row2, start2 = row + dr * n, start + dc * n
        dst = cells_of(row2, start2, L)
        for i in range(n):
            ops.append(DIR[d]); sels.append(act(src) if i == 0 else sel_of([]))
        for p in src:
            G[p] = 0
        for p in dst:
            G[p] = col
        vacated.update(set(src) - set(dst))
        vacated.difference_update(dst)
        bars[col] = (row2, start2, L)

    # colour 0 is "nothing" to ARCLE's object ops, so a 0-bar cannot be grabbed:
    # it is placed by paint once every other bar has settled (step 3).
    zero_bar = bars.pop(0, None)

    # 1. gravity inside each line: slide every bar flush against the far side
    for col in sorted(bars, key=lambda k: bars[k][0]):
        row, start, L = bars[col]
        n = W - L - start
        if n > 0:
            slide(col, 'R', n)

    # 2. sort the lines: stack bars by length against the wall, longest nearest
    all_len = dict((k, v[2]) for k, v in bars.items())
    if zero_bar is not None:
        all_len[0] = zero_bar[2]
    full_order = sorted(all_len, key=lambda k: (-all_len[k], k))
    slot_all = {col: H - 2 - i for i, col in enumerate(full_order)}
    order = [k for k in full_order if k != 0]
    slot = {k: slot_all[k] for k in order}
    done = set()

    def occupant(row):
        for k in bars:
            if k not in done and bars[k][0] == row:
                return k
        return None

    def vmove(col, row):
        r0 = bars[col][0]
        if row != r0:
            slide(col, 'D' if row > r0 else 'U', abs(row - r0))

    guard = 0
    while len(done) < len(order):
        guard += 1
        if guard > 100:
            raise ValueError("deadlock")
        progressed = False
        for col in order:
            if col in done:
                continue
            o = occupant(slot[col])
            if o is None or o == col:
                vmove(col, slot[col])
                done.add(col)
                progressed = True
                break
        if progressed:
            continue
        used = {bars[k][0] for k in bars if k not in done} | {slot[k] for k in done}
        if zero_bar is not None:
            used.add(slot_all[0])
        free_rows = [r for r in range(H - 1) if r not in used]
        if not free_rows:
            raise ValueError("no free row")
        col = next(k for k in order if k not in done)
        blk = occupant(slot[col])
        r0 = bars[blk][0]
        park = min(free_rows, key=lambda r: abs(r - r0))
        vmove(blk, park)

    # 3. the 0-bar: draw it right-aligned in its slot (cells not already 0);
    #    its old cells become an ordinary vacated footprint
    if zero_bar is not None:
        row, start, L = zero_bar
        src = cells_of(row, start, L)
        dst = cells_of(slot_all[0], W - L, L)
        new = [p for p in dst if G[p] != 0]
        if new:
            ops.append(0); sels.append(act(new))
        for p in dst:
            G[p] = 0
        vacated.update(p for p in src if p not in dst and G[p] == 0)
        vacated.difference_update(dst)

    # 4. the footprints the bars left behind read 0: restore them to background
    hole = sorted(p for p in vacated if G[p] == 0)   # vacated and not since covered
    if bgc != 0 and hole:
        ops.append(bgc); sels.append(act(hole))

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
