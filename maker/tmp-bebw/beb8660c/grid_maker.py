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
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]  # number of CW quarter turns applied to the canonical (8-at-bottom) layout


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = list(ROTS) + [random.choice(ROTS) for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = random.sample(ROTS, n_ex)
    plan = [{"rot": r} for r in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None, **kwargs) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    cols = remove(8, interval(0, 10, 1))
    lim = min(max_h, max_w, 30)
    w = unifint(diff_lb, diff_ub, (3, lim))
    h = unifint(diff_lb, diff_ub, (w, lim))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        j = h - k - 1 + j
        gi = fill(gi, c, connect((j, 0), (j, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    if len(gi) > max_h or len(gi[0]) > max_w:
        raise ValueError("too big")
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Each coloured bar is an object: slide it (unit Moves, one grab per bar)
    from where it lies in I to its slot in the stack against the 8-wall,
    longest bar next to the wall, all bars flush with the corner that sits
    counter-clockwise of the wall.  Everything is measured from I alone."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- the 8 wall ---------------------------------------------------------
    edge = None
    if np.all(I[0, :] == 8) and hi > 1 and not np.all(I[:, 0] == 8):
        edge = 'top'
    elif np.all(I[hi - 1, :] == 8) and not np.all(I[:, wi - 1] == 8):
        edge = 'bottom'
    elif np.all(I[:, 0] == 8):
        edge = 'left'
    elif np.all(I[:, wi - 1] == 8):
        edge = 'right'
    if edge is None:
        for name, line in (('top', I[0, :]), ('bottom', I[hi - 1, :]),
                           ('left', I[:, 0]), ('right', I[:, wi - 1])):
            if np.all(line == 8):
                edge = name
                break
    horizontal = edge in ('top', 'bottom')

    # --- background: the one colour present in every lane parallel to the wall
    if horizontal:
        lanes = [I[r, :] for r in range(hi) if not np.all(I[r, :] == 8)]
    else:
        lanes = [I[:, c] for c in range(wi) if not np.all(I[:, c] == 8)]
    common = set(range(10))
    for ln in lanes:
        common &= set(ln.tolist())
    common.discard(8)
    if len(common) == 1:
        bgc = common.pop()
    else:
        vals, cnts = np.unique(I[I != 8], return_counts=True)
        bgc = int(vals[np.argmax(cnts)])

    # --- bars (one per colour) ----------------------------------------------
    bars = {}
    for col in sorted(set(I.flatten().tolist()) - {bgc, 8}):
        cells = [tuple(map(int, x)) for x in np.argwhere(I == col)]
        bars[col] = cells
    order = sorted(bars, key=lambda c: (-len(bars[c]), c))

    dest = {}
    for i, col in enumerate(order):
        L = len(bars[col])
        if edge == 'top':
            dest[col] = [(1 + i, c) for c in range(L)]
        elif edge == 'bottom':
            dest[col] = [(hi - 2 - i, c) for c in range(wi - L, wi)]
        elif edge == 'left':
            dest[col] = [(r, 1 + i) for r in range(hi - L, hi)]
        else:
            dest[col] = [(r, wi - 2 - i) for r in range(L)]

    G = I.copy()          # tracks the visible working grid
    pos = {c: list(bars[c]) for c in bars}

    def is_free(cells, own):
        own = set(own)
        return all(G[r, c] == bgc or (r, c) in own for r, c in cells)

    def slide(col, target):
        """Translate bar `col` from pos[col] to `target` by unit Moves."""
        src = pos[col]
        dr = target[0][0] - min(r for r, _ in src)
        dc = target[0][1] - min(c for _, c in src)
        if horizontal:     # cross the lanes first, then run along the bar
            steps = [(1 if dr > 0 else -1, 0)] * abs(dr) + [(0, 1 if dc > 0 else -1)] * abs(dc)
        else:
            steps = [(0, 1 if dc > 0 else -1)] * abs(dc) + [(1 if dr > 0 else -1, 0)] * abs(dr)
        if not steps:
            return
        cur = list(src)
        if col == 0:
            # ARCLE's Move cannot carry colour 0 (it pastes only non-zero
            # cells), so a black bar is lifted and laid down by paint.
            ops.append(bgc); sels.append(sel_of(src))
            for r, c in src:
                G[r, c] = bgc
            cur = [(r + dr, c + dc) for r, c in src]
            ops.append(0); sels.append(sel_of(cur))
            for r, c in cur:
                G[r, c] = 0
            pos[col] = cur
            return
        snap = G.copy()
        for r, c in src:
            snap[r, c] = 0
        first = True
        for (sr, sc) in steps:
            ops.append({(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(sr, sc)])
            sels.append(sel_of(cur) if first else sel_of([]))
            first = False
            cur = [(r + sr, c + sc) for r, c in cur]
        G[:, :] = snap
        for r, c in cur:
            G[r, c] = col
        hole = sorted(set(src) - set(cur))
        if bgc != 0 and hole:
            ops.append(bgc); sels.append(sel_of(hole))
            for r, c in hole:
                G[r, c] = bgc
        pos[col] = cur

    remaining = list(order)
    lifted = []
    while remaining:
        ready = [c for c in remaining if is_free(dest[c], pos[c])]
        if ready:
            col = ready[0]
            slide(col, dest[col])
            remaining.remove(col)
            continue
        # every remaining bar's slot is blocked by another remaining bar:
        # park one blocking bar on a free stretch that blocks no slot.
        blocked_slots = set()
        for c in remaining:
            blocked_slots |= set(dest[c])
        parked = False
        for col in remaining:
            L = len(pos[col])
            cands = []
            if horizontal:
                for r in range(hi):
                    for c0 in range(wi - L + 1):
                        cands.append([(r, c0 + t) for t in range(L)])
            else:
                for c in range(wi):
                    for r0 in range(hi - L + 1):
                        cands.append([(r0 + t, c) for t in range(L)])
            sr0, sc0 = min(pos[col])
            cands.sort(key=lambda cs: abs(cs[0][0] - sr0) + abs(cs[0][1] - sc0))
            for cs in cands:
                if set(cs) == set(pos[col]):
                    continue
                if not is_free(cs, pos[col]):
                    continue
                if set(cs) & blocked_slots:
                    continue
                slide(col, cs)
                parked = True
                break
            if parked:
                break
        if not parked:
            # no parking room at all: lift one bar off (erase it) and lay it
            # into its slot once the others have cleared the way.
            col = remaining[0]
            ops.append(bgc); sels.append(sel_of(pos[col]))
            for r, c in pos[col]:
                G[r, c] = bgc
            remaining.remove(col)
            lifted.append(col)
    for col in lifted:
        ops.append(col); sels.append(sel_of(dest[col]))
        for r, c in dest[col]:
            G[r, c] = col

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
