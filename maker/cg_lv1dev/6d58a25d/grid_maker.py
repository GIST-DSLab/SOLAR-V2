"""
ARC Task: 6d58a25d (RE-ARC) — LLM-generated grid_maker
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
    rem = [c for c in cols if c != bgc]
    c1 = random.choice(rem)
    c2 = random.choice([c for c in rem if c != c1])
    return {"bgc": bgc, "c1": c1, "c2": c2}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    base = [(0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)]
    shp = [(r, c + 3) for r, c in base]
    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    loci = random.randint(0, h - 4)
    locj = random.randint(0, w - 7)
    plcd = set((r + loci, c + locj) for r, c in shp)
    rem = [(i, j) for i in range(h) for j in range(w) if (i, j) not in plcd]
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = random.sample(rem, nnoise)
    gi = np.full((h, w), bgc, dtype=int)
    for r, c in nois:
        gi[r, c] = c2
    for r, c in plcd:
        gi[r, c] = c1
    go = gi.copy()
    for (i, j) in nois:
        colcells = [r for (r, c) in plcd if c == j]
        if colcells and min(colcells) < i:
            for r in range(max(colcells) + 1, h):
                go[r, j] = c2
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    Reading: the maker and the code-golf reference agree, so the carving is unchanged.
    Turn the V template to its canonical pose (apex up). Each template column is one
    emitter. A column fires if a noise cell sits below the template in that column.
    It then sends ONE solid ray of the noise colour from just past the template's last
    cell in that column to the grid edge.
    The golf code checks for noise below the LAST template cell. The maker checks
    below the FIRST. These give the same answer because each template column is
    contiguous. Both treat a column's ray as one unit, so we use one Color op per
    firing column (the selection is the ray, named from I alone), in template order.
    """
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    base = [(0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)]
    canon = sorted((r, c + 3) for r, c in base)

    def norm(cells):
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        return sorted((r - r0, c - c0) for r, c in cells)

    ops, sels = [], []
    # background: generator keeps noise below half of the free cells -> majority colour
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    idx = np.arange(hi * wi).reshape(hi, wi)

    found = None
    for k in range(4):
        J = np.rot90(I, k)
        Jidx = np.rot90(idx, k)
        for col in sorted(set(J.flatten().tolist()) - {bgc}):
            cells = [(int(r), int(c)) for r, c in zip(*np.where(J == col))]
            if len(cells) == len(canon) and norm(cells) == canon:
                found = (J, Jidx, col, cells)
                break
        if found:
            break
    if found is None:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    J, Jidx, tcol, tcells = found
    hJ = J.shape[0]
    others = sorted(set(J.flatten().tolist()) - {bgc, tcol})
    if not others:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels
    ncol = others[0]

    by_col = {}
    for r, c in tcells:
        by_col.setdefault(c, []).append(r)

    # walk the template's columns in order; each firing column = one ray = one Color op
    for j in sorted(by_col):
        last = max(by_col[j])
        if not any(J[i, j] == ncol for i in range(last + 1, hJ)):
            continue
        ray = [(i, j) for i in range(last + 1, hJ)]
        if not ray or all(J[i, jj] == ncol for i, jj in ray):
            continue  # ray already fully present -> op would be a no-op
        ops.append(int(ncol))
        sels.append(sel_of([divmod(int(Jidx[r, c]), wi) for r, c in ray]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6d58a25d"
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
                                f"for task 6d58a25d"
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
                    f"Failed to build a complete episode for task 6d58a25d "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6d58a25d-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
