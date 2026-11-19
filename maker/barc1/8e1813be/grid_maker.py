"""
ARC Task: 8e1813be (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    return min(b, max(a, a + round(d * (b - a))))


def sample_colors(num_examples=None) -> dict:
    bgc, sqc = random.sample(range(10), 2)
    n_ex = num_examples if num_examples else 3
    V = [{"transpose": False}, {"transpose": True}]
    if n_ex >= 2:
        ex = [dict(v) for v in V] + [dict(random.choice(V)) for _ in range(n_ex - 2)]
        random.shuffle(ex)
    else:
        ex = [dict(random.choice(V))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": ex + [dict(random.choice(ex))]}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=5, transpose=None, **kw) -> dict:
    if transpose is None:
        transpose = random.choice([False, True])
    limH, limW = (max_w, max_h) if transpose else (max_h, max_w)
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    nub = min(8, limW - 3, limH // 3)
    if nub < 3:
        raise ValueError("grid too small")
    nbars = _unifint(diff_lb, diff_ub, (3, nub))
    ccols = random.sample(remcols, nbars)
    w = _unifint(diff_lb, diff_ub, (nbars + 3, limW))
    hmarg = _unifint(diff_lb, diff_ub, (2 * nbars, limH - nbars))
    gi = [[c] * w for c in ccols]
    go = [[c] * nbars for c in ccols]
    for _ in range(hmarg):
        idx = random.randint(0, len(gi) - 1)
        gi = gi[:idx] + [[bgc] * w] + gi[idx:]
    h2 = nbars + hmarg
    n = nbars
    loci = random.randint(1, h2 - n - 2)
    locj = random.randint(1, w - n - 2)
    gi = np.array(gi, dtype=int)
    gi[loci - 1:loci + n + 1, locj - 1:locj + n + 1] = bgc   # outbox -> bgc
    gi[loci:loci + n, locj:locj + n] = sqc                    # the square
    go = np.array(go, dtype=int)
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    """
    Reading: I holds several 1-thick stripes (each colour confined to one row or one
    column) and a solid square whose side equals the number of stripes. Fill the square,
    line by line, with the stripe colours in the order and direction they appear; the
    filled square is the answer.
    Ops: one Color op per stripe, painting that stripe's colour onto its line of the
    square (a full row / column of the square, in stripe order), then crop to the square.
    """
    I = np.asarray(I, dtype=int)
    bars_h, bars_v, amb, blocks = [], [], [], []
    for v in np.unique(I):
        rs, cs = np.where(I == v)
        R, C = sorted(set(rs.tolist())), sorted(set(cs.tolist()))
        if len(R) == 1 and len(C) == 1:
            amb.append((R[0], C[0], int(v)))       # stripe cut down to one cell
        elif len(R) == 1:
            bars_h.append((R[0], int(v)))
        elif len(C) == 1:
            bars_v.append((C[0], int(v)))
        else:
            r0, r1, c0, c1 = R[0], R[-1], C[0], C[-1]
            if r1 - r0 == c1 - c0 and len(rs) == (r1 - r0 + 1) * (c1 - c0 + 1):
                blocks.append((r0, c0, r1 - r0 + 1))   # solid square candidate
    vertical = len(bars_v) > len(bars_h)
    if vertical:
        bars = bars_v + [(c, v) for (r, c, v) in amb]
    else:
        bars = bars_h + [(r, v) for (r, c, v) in amb]
    bars.sort()                                       # order of appearance
    n = len(bars)
    sq = [b for b in blocks if b[2] == n]             # the square: side == #stripes
    sr, sc, _ = sq[0]

    ops, sels = [], []
    for k, (_, col) in enumerate(bars):
        if vertical:
            cells = [(sr + i, sc + k) for i in range(n)]   # column k of the square
        else:
            cells = [(sr + k, sc + j) for j in range(n)]   # row k of the square
        ops.append(int(col))
        sels.append(sel_of(cells))

    # crop the canvas to the (now filled) square -- a full rectangle
    ops.append(33)
    sels.append([sr, sc, n - 1, n - 1])
    ops.append(34)
    sels.append([0, 0, n - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 8e1813be"
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
                                f"for task 8e1813be"
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
                    f"Failed to build a complete episode for task 8e1813be "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"8e1813be-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
