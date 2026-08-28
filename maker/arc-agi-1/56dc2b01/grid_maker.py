"""
ARC Task: 56dc2b01 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = int(a + diff_lb * (b - a))
    hi = int(a + diff_ub * (b - a))
    lo = min(max(a, lo), b)
    hi = min(max(a, hi), b)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


# The four structural variants: where the 2-bar lies, and which side of it the
# object sits on (i.e. which way the object has to slide).
VARIANTS = [
    {"orientation": "vertical",   "mirrored": False},   # bar |, object right of it
    {"orientation": "vertical",   "mirrored": True},    # bar |, object left of it
    {"orientation": "horizontal", "mirrored": False},   # bar -, object below it
    {"orientation": "horizontal", "mirrored": True},    # bar -, object above it
]


def sample_colors(num_examples=None) -> dict:
    # objc must be non-zero: ARCLE object ops (Move) only grab non-zero cells.
    objc = random.choice([c for c in range(1, 10) if c not in (2, 8)])
    bgc = random.choice([c for c in range(10) if c not in (2, 8, objc)])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc,
             orientation=None, mirrored=None) -> dict:
    if orientation is None:
        orientation = random.choice(["vertical", "horizontal"])
    if mirrored is None:
        mirrored = random.choice([True, False])

    max_h = min(int(max_h), 30)
    max_w = min(int(max_w), 30)
    if orientation == "horizontal" and (max_w < 4 or max_h < 6):
        orientation = "vertical"
    # base frame: bar is a vertical line, object to its right
    if orientation == "vertical":
        H, W = max_h, max_w
    else:
        H, W = max_w, max_h          # grid gets transposed at the end

    h = _unifint(diff_lb, diff_ub, (4, H))
    w = _unifint(diff_lb, diff_ub, (6, W))

    oh = _unifint(diff_lb, diff_ub, (1, h))
    ow = _unifint(diff_lb, diff_ub, (1, max(1, (w - 1) // 2 - 1)))

    # ---- grow a connected (8-neighbour) blob inside an oh x ow box ----
    sp = (random.randrange(oh), random.randrange(ow))
    obj = set()
    cand = set()

    def _add(cell):
        obj.add(cell)
        cand.discard(cell)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (cell[0] + dr, cell[1] + dc)
                if 0 <= n[0] < oh and 0 <= n[1] < ow and n not in obj:
                    cand.add(n)

    _add(sp)
    ncellsd = _unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = random.choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for _ in range(ncells):
        if not cand:
            break
        _add(random.choice(sorted(cand)))

    mi = min(r for r, _ in obj)
    mj = min(c for _, c in obj)
    obj = {(r - mi, c - mj) for r, c in obj}
    oh = max(r for r, _ in obj) + 1
    ow = max(c for _, c in obj) + 1

    # ---- placement ----
    loci = random.randint(0, h - oh)
    locj = _unifint(diff_lb, diff_ub, (2, w - ow))          # >=2 -> object really slides
    barlocji = _unifint(diff_lb, diff_ub, (0, locj))
    ub = min(locj - 2, w - 2 - ow)                          # keeps the 8-line inside the grid
    barlocj = min(max(0, locj - barlocji), ub)

    gi = [[bgc] * w for _ in range(h)]
    for r in range(h):
        gi[r][barlocj] = 2
    go = [row[:] for row in gi]
    for (r, c) in obj:
        go[loci + r][barlocj + 1 + c] = objc                # slid against the bar
    for r in range(h):
        go[r][barlocj + ow + 1] = 8                         # line beyond the far edge
    for (r, c) in obj:
        gi[loci + r][locj + c] = objc

    gi = np.array(gi, dtype=int)
    go = np.array(go, dtype=int)
    if orientation == "horizontal":
        gi, go = gi.T.copy(), go.T.copy()                   # dmirror
        if mirrored:
            gi, go = np.flipud(gi).copy(), np.flipud(go).copy()
    else:
        if mirrored:
            gi, go = np.fliplr(gi).copy(), np.fliplr(go).copy()

    return {"input": gi.tolist(), "output": go.tolist()}


# ------------------------------------------------------------- derivation ---
def _transpose_ops(h, w):
    """Ops performing dmirror (transpose) of the whole h x w working grid.

    transpose == flipud(rot90_CCW(grid)).  Every selection here is a FULL
    rectangle (background included) on purpose - the reflection applies to the
    whole canvas.
    """
    ops, sels = [], []
    sq = max(h, w)
    if h == w:
        full = [0, 0, sq - 1, sq - 1]
        ops.append(24); sels.append(full)                       # whole grid, CCW
        ops.append(27); sels.append(full)                       # whole grid, up<->down
    else:
        ops.append(33); sels.append([0, 0, sq - 1, sq - 1])     # square the canvas
        ops.append(24); sels.append([0, 0, sq - 1, sq - 1])     # CCW on the square
        ops.append(27); sels.append([sq - w, 0, w - 1, h - 1])  # flip the content block
        ops.append(33); sels.append([sq - w, 0, w - 1, h - 1])  # crop back to w x h
    return ops, sels


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # The 2-line is the wall.  If it lies along a row, mirror the grid across its
    # diagonal so the wall stands vertical (exactly the verifier's dmirror branch).
    bar = np.argwhere(I == 2)
    horizontal = len(set(int(r) for r, _ in bar)) == 1

    if horizontal:
        t_ops, t_sels = _transpose_ops(hi, wi)
        ops += t_ops; sels += t_sels
        G = I.T.copy()
    else:
        G = I.copy()

    h, w = G.shape
    cnt = Counter(G.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    objc = [c for c in cnt if c != bgc and c != 2][0]
    bc = int(np.argwhere(G == 2)[0][1])

    obj = [(int(r), int(c)) for r, c in np.argwhere(G == objc)]
    cmin = min(c for _, c in obj)
    cmax = max(c for _, c in obj)
    ow = cmax - cmin + 1

    if cmin > bc:                      # object right of the wall -> slide left
        dc = (bc + 1) - cmin
        line_col = bc + ow + 1
    else:                              # object left of the wall -> slide right
        dc = (bc - 1) - cmax
        line_col = bc - ow - 1

    # slide the object until it touches the wall: one grab, then empty selections
    step = 1 if dc > 0 else -1
    move_op = 22 if dc > 0 else 23
    cur = list(obj)
    for k in range(abs(dc)):
        ops.append(move_op)
        sels.append(sel_of(cur) if k == 0 else sel_of([]))
        cur = [(r, c + step) for r, c in cur]

    # only the footprint the object left behind needs repainting
    hole = sorted(set(obj) - set(cur))
    if bgc != 0 and hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))

    # the 8-line: full line, one cell beyond the object's far edge
    ops.append(8)
    sels.append(sel_of([(r, line_col) for r in range(h)]))

    if horizontal:
        t_ops, t_sels = _transpose_ops(h, w)     # mirror back across the diagonal
        ops += t_ops; sels += t_sels

    ho, wo = O.shape
    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 56dc2b01"
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
                                f"for task 56dc2b01"
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
                    f"Failed to build a complete episode for task 56dc2b01 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"56dc2b01-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
