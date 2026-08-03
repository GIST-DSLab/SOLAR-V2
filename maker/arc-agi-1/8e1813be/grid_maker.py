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
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    sqc = random.choice([c for c in cols if c != bgc])
    # discrete structural variant: bars run as rows (mirror=False) or as columns (mirror=True)
    VARIANTS = [{"mirror": False}, {"mirror": True}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sqc": sqc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, sqc: int, mirror=None) -> dict:
    if mirror is None:
        mirror = choice((True, False))
    cols = interval(0, 10, 1)
    remcols = remove(bgc, remove(sqc, cols))

    # h2 = nbars + hmarg rows, w cols before the optional dmirror (which transposes)
    hlim = min(30, max_w if mirror else max_h)
    wlim = min(30, max_h if mirror else max_w)

    nb_ub = min(8, hlim // 3, wlim - 3)
    nb_ub = max(3, nb_ub)
    nbars = unifint(diff_lb, diff_ub, (3, nb_ub))
    ccols = sample(remcols, nbars)
    w = unifint(diff_lb, diff_ub, (nbars + 3, max(nbars + 3, wlim)))
    hmarg_ub = max(2 * nbars, min(30 - nbars, hlim - nbars))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, hmarg_ub))

    ccols = list(ccols)
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = randint(0, len(go) - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = randint(1, h2 - oh - 2)
    locj = randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I):
      I holds several 1-thick bars (each bar = every cell of one colour lies in a single
      row, or a single column), plus a fat square block and background.  The answer is those
      bars alone, squeezed together side by side, keeping their original order and as many
      lines as there are bars.
    Ops: for each bar, in bar order, paint its colour onto its slot of the n*n corner block
         (skipped when the bar already occupies that slot), then crop to that block.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # --- find the 1-thick bars in I (colour confined to one row / one column) -------------
    bars_h, bars_v, amb = [], [], []          # (row, colour) / (col, colour) / ambiguous
    for v in np.unique(I):
        rs, cs = np.where(I == v)
        nr, nc = len(set(rs.tolist())), len(set(cs.tolist()))
        if nr == 1 and nc == 1:               # bar eaten down to a single cell by the square
            amb.append((int(rs[0]), int(cs[0]), int(v)))
        elif nr == 1:
            bars_h.append((int(rs[0]), int(v)))
        elif nc == 1:
            bars_v.append((int(cs[0]), int(v)))

    vertical = len(bars_v) > len(bars_h)      # orientation given by the full-length bars
    if vertical:
        bars = bars_v + [(c, v) for (r, c, v) in amb]
    else:
        bars = bars_h + [(r, v) for (r, c, v) in amb]
    bars.sort()                               # order = leftmost (or uppermost) in I
    n = len(bars)

    ops, sels = [], []

    # --- lay the bars down, one op per bar object, in bar order ---------------------------
    for k, (pos, col) in enumerate(bars):
        if vertical:
            if not bool(np.all(I[0:n, k] == col)):   # bar k already sits in slot k -> nothing to do
                ops.append(int(col))
                sels.append([0, k, n - 1, 0])
        else:
            if not bool(np.all(I[k, 0:n] == col)):
                ops.append(int(col))
                sels.append([k, 0, 0, n - 1])

    # --- keep only the assembled block (bars no longer needed) ----------------------------
    ops.append(33)
    sels.append([0, 0, n - 1, n - 1])
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
