"""
ARC Task: d4469b4b (RE-ARC) — LLM-generated grid_maker
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

KEY_COLS = [1, 2, 3]
# generator: col 2 -> template A (cross centre (0,1)), col 1 -> B (1,1), col 3 -> C (2,2)
DEFAULT_CENTRE = {2: (0, 1), 1: (1, 1), 3: (2, 2)}


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(KEY_COLS):
        examples = list(KEY_COLS) + [random.choice(KEY_COLS) for _ in range(n_ex - len(KEY_COLS))]
        random.shuffle(examples)
    else:
        examples = random.sample(KEY_COLS, n_ex)
    plan = [{"col": c} for c in examples]
    plan.append({"col": random.choice(examples)})
    return {"instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, col=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    def make(cells):
        g = [[5] * 3 for _ in range(3)]
        for r, c in cells:
            g[r][c] = 0
        return g

    A = make({(1, 0), (2, 0), (1, 2), (2, 2)})
    B = make({(0, 0), (0, 2), (2, 0), (2, 2)})
    C = make({(0, 0), (0, 1), (1, 0), (1, 1)})
    table = {2: A, 1: B, 3: C}
    if col is None:
        col = random.choice(KEY_COLS)
    go = [row[:] for row in table[col]]

    cols = [c for c in range(10) if c not in (1, 2, 3)]
    h = max(2, min(max_h, unifint(diff_lb, diff_ub, (2, max_h))))
    w = max(2, min(max_w, unifint(diff_lb, diff_ub, (2, max_w))))
    gi = [[col] * w for _ in range(h)]
    numc = unifint(diff_lb, diff_ub, (1, 7))
    numc = max(1, min(7, numc))
    ccols = random.sample(cols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    numcells = unifint(diff_lb, diff_ub, (0, h * w - 1))
    numcells = max(0, min(h * w - 1, numcells))
    for (i, j) in random.sample(inds, numcells):
        gi[i][j] = random.choice(ccols)
    return {"input": gi, "output": go}


def _key_colour(G):
    present = set(int(v) for v in np.unique(G))
    key = sorted(present & set(KEY_COLS))
    return key[0] if key else None


def _cross_centre(O):
    O = np.asarray(O, dtype=int)
    rr = [r for r in range(3) if all(O[r, c] == 5 for c in range(3))]
    cc = [c for c in range(3) if all(O[r, c] == 5 for r in range(3))]
    if rr and cc:
        return rr[0], cc[0]
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)

    # Rule learned from the demonstrations: the canvas colour (the one key colour
    # in {1,2,3}; noise never uses these) determines where the 5-cross sits.
    learned = {}
    for ex in (examples or []):
        try:
            ei, eo = ex
        except Exception:
            ei, eo = ex["input"], ex["output"]
        k = _key_colour(np.asarray(ei, dtype=int))
        ctr = _cross_centre(eo)
        if k is not None and ctr is not None:
            learned.setdefault(k, ctr)

    key = _key_colour(I)
    if key is None:
        key = 1
    cr, cc = learned.get(key, DEFAULT_CENTRE[key])

    ops, sels = [], []
    # 1. shrink the canvas to the 3x3 answer board (full rectangle)
    ops.append(33); sels.append([0, 0, 2, 2])
    # 2. blank the board: crop keeps the input's top-left 3x3 nonzero pixels,
    #    which carry no meaning for the answer (skip if already all 0)
    if np.any(I[:3, :3] != 0):
        ops.append(0); sels.append([0, 0, 2, 2])
    # 3. draw the cross: full row through the centre, then full column
    ops.append(5); sels.append(sel_of([(cr, c) for c in range(3)]))
    ops.append(5); sels.append(sel_of([(r, cc) for r in range(3)]))
    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task d4469b4b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d4469b4b"
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
                                f"for task d4469b4b"
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
                    f"Failed to build a complete episode for task d4469b4b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d4469b4b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
