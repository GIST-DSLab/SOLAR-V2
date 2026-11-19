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
    cols = remove(8, interval(0, 10, 1))
    if bgc is None:
        bgc = choice(cols)
    if rot is None:
        rot = choice(ROTS)
    w_hi = min(30, max_w if rot in (0, 2) else max_h)
    h_hi = min(30, max_h if rot in (0, 2) else max_w)
    w = unifint(diff_lb, diff_ub, (3, w_hi))
    h = unifint(diff_lb, diff_ub, (w, max(w, h_hi)))
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
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Golf reading sorted(map(sorted, g)) in the canonical frame (8-line at the
    bottom): every bar is pushed flush to the far side of its own lane (sort each
    row) and the lanes are reordered by bar length, longest next to the 8-line
    (sort the rows).  Each bar is one object, translated with a single grabbed
    Move chain from wherever it lies to its slot."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # which edge holds the 8-line -> map canonical (r, c) to actual cells
    if np.all(I[hi - 1, :] == 8):
        H, W = hi, wi; f = lambda r, c: (r, c); park = H; horiz = True
    elif np.all(I[0, :] == 8):
        H, W = hi, wi; f = lambda r, c: (H - 1 - r, W - 1 - c); park = -1; horiz = True
    elif np.all(I[:, 0] == 8):
        H, W = wi, hi; f = lambda r, c: (c, H - 1 - r); park = -1; horiz = False
    else:
        H, W = wi, hi; f = lambda r, c: (W - 1 - c, r); park = H; horiz = False

    C = np.array([[I[f(r, c)] for c in range(W)] for r in range(H)])
    # background: the only colour (besides 8) present in every lane
    common = set(C[0].tolist())
    for r in range(1, H - 1):
        common &= set(C[r].tolist())
    common.discard(8)
    if len(common) == 1:
        bgc = common.pop()
    else:
        bgc = Counter(C[:H - 1].flatten().tolist()).most_common(1)[0][0]

    bars = []
    for r in range(H - 1):
        cs = [c for c in range(W) if C[r, c] != bgc]
        if cs:
            bars.append({"col": int(C[r, cs[0]]), "r": r, "c0": cs[0], "L": len(cs)})
    bars.sort(key=lambda b: -b["L"])
    for i, b in enumerate(bars):
        b["d"] = H - 2 - i            # longest bar right next to the 8-line
        b["dc"] = W - b["L"]          # flush against the far side of the lane

    def cells_at(b, r, c0):
        return {(r, c0 + j) for j in range(b["L"])}

    def cur(b):
        return cells_at(b, b["r"], b["c0"])

    def mop(dr, dc):
        (a, b_), (x, y) = f(0, 0), f(dr, dc)
        return {(-1, 0): 20, (1, 0): 21, (0, 1): 22, (0, -1): 23}[(x - a, y - b_)]

    G = {(r, c): int(C[r, c]) for r in range(H) for c in range(W)}  # canonical working grid
    state = {"expanded": False}

    def relocate(b, r, c0):
        src, dst = cur(b), cells_at(b, r, c0)
        if b["col"] == 0:
            # colour 0 is "nothing" to ARCLE's object buffer, so Move cannot carry
            # this bar: draw it at its new place; its old cells are left as 0 like
            # the footprint every Move leaves, and are cleared with those below.
            # (if a lifted bar already left exactly those cells at 0, they need no paint)
            if any(G[p] != 0 for p in dst):
                ops.append(0); sels.append(sel_of([f(*p) for p in sorted(dst)]))
        else:
            first = True
            for dr, dc, n in ((0, 1 if c0 > b["c0"] else -1, abs(c0 - b["c0"])),
                              (1 if r > b["r"] else -1, 0, abs(r - b["r"]))):
                for _ in range(n):
                    ops.append(mop(dr, dc))
                    sels.append(sel_of([f(*p) for p in sorted(src)]) if first else sel_of([]))
                    first = False
            for p in src:
                G[p] = 0                 # grabbed footprint reads 0 afterwards
        for p in dst:
            G[p] = b["col"]
        b["r"], b["c0"] = r, c0

    def free(b, cellset, strict=False):
        # a colour-0 bar is redrawn by paint, never lifted, so a bar settling for
        # good on it loses nothing; every other bar must not be covered before it
        # has been moved.  A bar only parked for a while (strict) keeps off the
        # colour-0 bar too, since lifting it again would leave that spot at 0.
        return all(not (cellset & cur(o)) for o in bars
                   if o is not b and (strict or o["col"] != 0))

    pending = [b for b in bars if (b["r"], b["c0"]) != (b["d"], b["dc"])]
    guard = 0
    while pending:
        guard += 1
        ready = [b for b in pending if free(b, cells_at(b, b["d"], b["dc"]))]
        if ready and guard < 400:
            b = ready[0]
            relocate(b, b["d"], b["dc"]); pending.remove(b)
            continue
        # every remaining bar is blocked: step a blocker aside
        a = pending[0]
        need = cells_at(a, a["d"], a["dc"])
        blockers = [o for o in pending if o is not a and (cur(o) & need)]
        bl = [o for o in blockers if o["col"] != 0] or blockers
        b = bl[0]
        dest_lanes = {o["d"] for o in pending}
        spot = None
        # 1) slide along its own lane out of the way
        for c0 in sorted(range(W - b["L"] + 1), key=lambda c: abs(c - b["c0"])):
            s = cells_at(b, b["r"], c0)
            if c0 != b["c0"] and not (s & need) and free(b, s, True):
                spot = (b["r"], c0); break
        # 2) a lane nobody still needs
        if spot is None:
            for r in sorted(range(H - 1), key=lambda r: abs(r - b["r"])):
                if r in dest_lanes or r == b["r"]:
                    continue
                for c0 in (W - b["L"],) + tuple(range(W - b["L"])):
                    s = cells_at(b, r, c0)
                    if free(b, s, True):
                        spot = (r, c0); break
                if spot:
                    break
        # 3) an extra parking lane outside the canvas
        if spot is None:
            if not state["expanded"]:
                ops.append(33)  # grow the canvas by one lane to park a bar
                sels.append([0, 0, hi, wi - 1] if horiz else [0, 0, hi - 1, wi])
                state["expanded"] = True
                for c in range(W):
                    G[(park, c)] = 0
            for c0 in [b["c0"]] + list(range(W - b["L"] + 1)):
                if free(b, cells_at(b, park, c0), True):
                    spot = (park, c0); break
            if spot is None:
                spot = (park, b["c0"])
        relocate(b, *spot)

    if state["expanded"]:
        ops.append(33); sels.append([0, 0, hi - 1, wi - 1])  # drop the parking lane

    # what is left at 0 is the footprint the bars were lifted from: give it back
    # the background (the colour-0 bar, if any, stays where it now stands)
    if bgc != 0:
        zero_bar = set()
        for b in bars:
            if b["col"] == 0:
                zero_bar |= cur(b)
        holes = [f(r, c) for r in range(H) for c in range(W)
                 if G[(r, c)] == 0 and (r, c) not in zero_bar]
        if holes:
            ops.append(int(bgc)); sels.append(sel_of(sorted(holes)))

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
