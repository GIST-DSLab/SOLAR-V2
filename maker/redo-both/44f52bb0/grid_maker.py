"""
ARC Task: 44f52bb0 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ----
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _identity(g):
    return [row[:] for row in g]


def _hmirror(g):                       # up <-> down
    return [row[:] for row in g[::-1]]


def _vmirror(g):                       # left <-> right
    return [row[::-1] for row in g]


def _dmirror(g):                       # transpose
    return [list(r) for r in zip(*g)]


def _rot180(g):
    return [row[::-1] for row in g[::-1]]


def _cmirror(g):                       # counterdiagonal mirror
    return _rot180(_dmirror(g))


def _rot90(g):                         # clockwise
    return [list(r) for r in zip(*g[::-1])]


def _rot270(g):                        # counter clockwise
    return [list(r) for r in zip(*g)][::-1]


def _is_vsym(g):
    return g == _vmirror(g)


def _is_hsym(g):
    return g == _hmirror(g)


# ------------------------------------------------------------ colour plan ----
VARIANTS = [{"issymm": True}, {"issymm": False}]


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# -------------------------------------------------------------- generate -----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, issymm=None) -> dict:
    if issymm is None:
        issymm = random.choice([True, False])

    h = _unifint(diff_lb, diff_ub, (3, max_h))
    w = _unifint(diff_lb, diff_ub, (3, max_w))

    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (2, 9))
    ccols = random.sample(remcols, ncols)

    numcells = _unifint(diff_lb, diff_ub, (1, h * w - 1))
    inds = [(i, j) for i in range(h) for j in range(w)]

    gi = [[bgc] * w for _ in range(h)]
    while _is_hsym(gi):
        cells = random.sample(inds, numcells)
        gi = [[bgc] * w for _ in range(h)]
        for (a, b) in cells:
            col = random.choice(ccols)
            gi[a][b] = col
            gi[a][w - 1 - b] = col
    # gi is now left-right symmetric but not up-down symmetric

    if not issymm:
        cands = [(i, j) for i in range(h) for j in range(w // 2)]
        pool = sorted(set(ccols) | {bgc})
        while True:
            gp = [row[:] for row in gi]
            numpert = _unifint(diff_lb, diff_ub, (1, h * (w // 2)))
            numpert = min(numpert, len(cands))
            locs = random.sample(cands, numpert)
            for a, b in locs:
                col = gp[a][b]
                choices = [c for c in pool if c != col]
                if choices:
                    gp[a][b] = random.choice(choices)
            if (not _is_vsym(gp)) and (not _is_hsym(gp)):
                gi = gp
                break
        go = [[7]]
    else:
        go = [[1]]

    all_mfs = [_identity, _dmirror, _cmirror, _vmirror, _hmirror, _rot90, _rot180, _rot270]
    keep_shape = [_identity, _vmirror, _hmirror, _rot180]
    transpose_ok = (h <= max_w and w <= max_h)
    mfs = all_mfs if transpose_ok else keep_shape
    nmfs = random.choice([1, 2])
    for fn in random.sample(mfs, min(nmfs, len(mfs))):
        gi = fn(gi)

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ------------------------------------------------------- derive_operations ---
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- the rule, measured from I alone -------------------------------------
    # mirror-symmetric (left<->right or up<->down)  -> 1 ,  otherwise -> 7
    v_sym = bool(np.array_equal(I, np.fliplr(I)))
    h_sym = bool(np.array_equal(I, np.flipud(I)))
    target = 1 if (v_sym or h_sym) else 7

    # --- pick the cell the 1x1 canvas is cut from ----------------------------
    # when the grid is NOT symmetric, cut it from a cell that actually witnesses
    # the broken symmetry (differs from its mirror partner); otherwise any cell.
    witnesses = []
    if target == 7:
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != I[r, wi - 1 - c] or I[r, c] != I[hi - 1 - r, c]:
                    witnesses.append((r, c))
    candidates = witnesses if witnesses else [(r, c) for r in range(hi) for c in range(wi)]

    cut = None
    for (r, c) in candidates:
        if I[r, c] != target:
            cut = (r, c)
            break
    if cut is None:
        for r in range(hi):
            for c in range(wi):
                if I[r, c] != target:
                    cut = (r, c)
                    break
            if cut is not None:
                break
    if cut is None:
        cut = (0, 0)

    ops, sels = [], []

    # 1. shrink the canvas down to that single cell (the answer is 1x1)
    ops.append(33)
    sels.append(sel_of([cut]))

    # 2. paint the verdict colour onto it (skipped only if it already holds it)
    if int(I[cut[0], cut[1]]) != target:
        ops.append(target)
        sels.append(sel_of([(0, 0)]))

    ops.append(34)
    sels.append(sel_of([(0, 0)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 44f52bb0"
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
                                f"for task 44f52bb0"
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
                    f"Failed to build a complete episode for task 44f52bb0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"44f52bb0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
