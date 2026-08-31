"""
ARC Task: 82819916 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    ass, bss = random.sample(rem, 2)

    VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ass": ass, "bss": bss, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ass, bss, rot=None) -> dict:
    if rot is None:
        rot = choice((0, 1, 2, 3))

    # rot90/rot270 transpose the canvas -> keep both dims inside both limits
    if rot in (1, 3):
        hmax = min(max_h, max_w)
        wmax = min(max_h, max_w)
    else:
        hmax = max_h
        wmax = max_w

    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, hmax))
    w = unifint(diff_lb, diff_ub, (5, wmax))
    remcols = remove(bgc, cols)

    itv = interval(0, w, 1)
    na = randint(2, w - 2)
    alocs = sample(itv, na)
    blocs = difference(itv, alocs)
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
        a_col, b_col = bss, ass
    else:
        a_col, b_col = ass, bss

    llocs = randint(0, h - 1)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, a_col, {(llocs, j) for j in alocs})
    gi = fill(gi, b_col, {(llocs, j) for j in blocs})

    numl = unifint(diff_lb, diff_ub, (1, max(1, (h - 1) // 2)))
    remlocs = remove(llocs, interval(0, h, 1))
    for k in range(numl):
        lloc = choice(remlocs)
        remlocs = remove(lloc, remlocs)
        a, b = sample(remcols, 2)
        gi = fill(gi, a, {(lloc, j) for j in alocs})
        gi = fill(gi, b, {(lloc, j) for j in blocs})

    cutoff = min(blocs) + 1
    go = tuple(e for e in gi)
    gi = fill(gi, bgc, backdrop(frozenset({(0, cutoff), (h - 1, w - 1)})))
    gi = fill(gi, a_col, {(llocs, j) for j in alocs})
    gi = fill(gi, b_col, {(llocs, j) for j in blocs})

    rotf = (identity, rot90, rot180, rot270)[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. ops
def derive_operations(I, O):
    """
    Rule (measured from I alone):
      * exactly one FULL line (row, or column after a 90/270 rotation) contains no
        background colour -> that is the KEY line.  It is bicoloured and partitions
        the perpendicular index range into class A and class B.
      * every other drawn line is TRUNCATED: only a contiguous stub survives, and that
        stub straddles the class boundary, so it exposes one seed cell of class A and
        one seed cell of class B.
      * each truncated line is extended: its class-A seed colour fills every class-A
        index, its class-B seed colour fills every class-B index.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- orientation: measured from I (is the key line a row or a column?)
    horiz = any(not (I[r] == bgc).any() for r in range(hi))
    A = I if horiz else I.T
    h, w = A.shape

    def to_grid(r, j):
        return (r, j) if horiz else (j, r)

    # ---- key line + class partition, read from I's key stripe
    key = next(r for r in range(h) if not (A[r] == bgc).any())
    kr = A[key]
    cA = int(kr[0])
    classA = [j for j in range(w) if int(kr[j]) == cA]
    classB = [j for j in range(w) if int(kr[j]) != cA]

    ops, sels = [], []

    # ---- extend each truncated line, one line (object) at a time, top to bottom
    for r in range(h):
        if r == key:
            continue
        stub = [j for j in range(w) if int(A[r, j]) != bgc]
        if not stub:
            continue
        stub_set = set(stub)
        aj = [j for j in stub if int(kr[j]) == cA]
        bj = [j for j in stub if int(kr[j]) != cA]
        if not aj or not bj:
            continue
        seed_a = int(A[r, aj[0]])          # colour this line assigns to class A
        seed_b = int(A[r, bj[0]])          # colour this line assigns to class B

        for cls, col in ((classA, seed_a), (classB, seed_b)):
            # skip stub cells: they already hold exactly this colour
            cells = [to_grid(r, j) for j in cls if j not in stub_set]
            if not cells:
                continue
            ops.append(col)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 82819916"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 82819916"
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
                                f"for task 82819916"
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
                    f"Failed to build a complete episode for task 82819916 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"82819916-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
