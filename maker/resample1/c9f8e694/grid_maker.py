"""
ARC Task: c9f8e694 (RE-ARC) — LLM-generated grid_maker
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

ROTS = [0, 1, 2, 3]  # number of clockwise quarter turns applied to the canonical layout (strip on the left)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    sqc = random.choice(range(1, 10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"sqc": sqc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, sqc=None, rot=None) -> dict:
    if sqc is None:
        sqc = random.choice(range(1, 10))
    if rot is None:
        rot = random.choice(ROTS)
    bgc = 0
    # after a 90/270 rotation the height and width trade places
    if rot in (1, 3):
        bh, bw = max_w, max_h
    else:
        bh, bw = max_h, max_w
    h = _unifint(diff_lb, diff_ub, (5, max(5, min(30, bh))))
    w = _unifint(diff_lb, diff_ub, (5, max(5, min(30, bw))))
    remcols = [c for c in range(1, 10) if c != sqc]
    nsq = _unifint(diff_lb, diff_ub, (1, 8))
    gir = np.zeros((h, w - 1), dtype=int)
    gil = [random.choice(remcols) for _ in range(h)]
    succ, fails, maxfails = 0, 0, nsq * 5
    while succ < nsq and fails < maxfails:
        loci = random.randint(0, h - 3)
        locj = random.randint(0, w - 3)
        lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
        locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
        if lock <= h - 1 and locl <= w - 2:
            gir[loci:lock + 1, locj:locl + 1] = sqc
            succ += 1
        else:
            fails += 1
    rows_with = set(np.nonzero(gir == sqc)[0].tolist())
    gil = [gil[i] if i in rows_with else bgc for i in range(h)]
    giro = gir.copy()
    for i in range(h):
        giro[i][gir[i] == sqc] = gil[i]
    gl = np.array(gil, dtype=int).reshape(h, 1)
    gi = np.hstack([gl, gir])
    go = np.hstack([gl, giro])
    gi = np.rot90(gi, -rot)
    go = np.rot90(go, -rot)
    return {"input": [list(map(int, r)) for r in gi],
            "output": [list(map(int, r)) for r in go]}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # The colour strip is the edge line with the most distinct values
    # (checked in the verifier's order: left, top, bottom, right).
    cands = [
        ("left", I[:, 0]),
        ("top", I[0, :]),
        ("bottom", I[-1, ::-1]),
        ("right", I[:, -1]),
    ]
    best = None
    bestn = -1
    for name, line in cands:
        n = len(set(line.tolist()))
        if n > bestn:
            bestn = n
            best = name

    ops, sels = [], []
    if best in ("left", "right"):
        sc = 0 if best == "left" else wi - 1
        for r in range(hi):
            c = int(I[r, sc])
            if c == 0:
                continue
            cells = [(r, cc) for cc in range(wi) if cc != sc and I[r, cc] != 0 and I[r, cc] != c]
            if cells:
                ops.append(c)
                sels.append(sel_of(cells))
    else:
        sr = 0 if best == "top" else hi - 1
        for cc in range(wi):
            c = int(I[sr, cc])
            if c == 0:
                continue
            cells = [(r, cc) for r in range(hi) if r != sr and I[r, cc] != 0 and I[r, cc] != c]
            if cells:
                ops.append(c)
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
                        f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
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
                                f"for task c9f8e694"
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
                    f"Failed to build a complete episode for task c9f8e694 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c9f8e694-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
