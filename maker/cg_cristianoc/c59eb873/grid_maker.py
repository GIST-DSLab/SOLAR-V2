"""
ARC Task: c59eb873 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # the rule (upscale x2) is colour-independent; only bgc is an episode role
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    from random import sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + (b - a) * lb
        hi_ = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi_)))))

    if bgc is None:
        bgc = random.choice(range(10))
    h = unifint(diff_lb, diff_ub, (1, max(1, min(15, max_h // 2))))
    w = unifint(diff_lb, diff_ub, (1, max(1, min(15, max_w // 2))))
    gi = [[bgc] * w for _ in range(h)]
    remcols = [c for c in range(10) if c != bgc]
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = [(i, j) for i in range(h) for j in range(w)]
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        num = min(num, len(inds))
        chos = sample(inds, num)
        for i, j in chos:
            gi[i][j] = col
        cs = set(chos)
        inds = [x for x in inds if x not in cs]
    go = [[v for v in row for _ in range(2)] for row in gi for _ in range(2)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading (separable stretch): every column is doubled, then every row is
    # doubled -- i.e. upscale by 2 done one axis at a time, line by line.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    H, W = 2 * hi, 2 * wi
    ops, sels = [], []

    # 1. grow canvas once to the doubled size (bbox = full target rectangle)
    ops.append(33); sels.append([0, 0, H - 1, W - 1])
    clip = [None]  # clipboard content currently held (shape + values)
    cur = np.zeros((H, W), dtype=int)
    cur[:hi, :wi] = I

    def duplicate_line(src_cells, targets):
        # src_cells: cells of the source line in the current grid
        # targets: lines (cell lists) the source line must be written onto
        vals = [int(cur[r, c]) for r, c in src_cells]
        # base layer: clear target lines whose stale non-zero content would
        # survive under a 0 of the line (Paste is transparent for 0)
        stale = []
        for tl in targets:
            for (r, c), v in zip(tl, vals):
                if v == 0 and cur[r, c] != 0:
                    stale.append(tl)
                    break
        if stale:
            cells = [rc for tl in stale for rc in tl]
            ops.append(0); sels.append(sel_of(cells))
            for r, c in cells:
                cur[r, c] = 0
        # pastes that visibly change something
        eff = [tl for tl in targets
               if any(v != 0 and cur[r, c] != v for (r, c), v in zip(tl, vals))]
        if not eff:
            return
        r0 = min(r for r, _ in src_cells); c0 = min(c for _, c in src_cells)
        r1 = max(r for r, _ in src_cells); c1 = max(c for _, c in src_cells)
        key = (r1 - r0, c1 - c0, tuple(vals))
        if clip[0] != key:  # clipboard does not already hold this line
            # bbox = exactly the whole source line (row or column)
            ops.append(29); sels.append([r0, c0, r1 - r0, c1 - c0])
            clip[0] = key
        for tl in eff:
            tr = min(r for r, _ in tl); tc = min(c for _, c in tl)
            ops.append(30); sels.append([tr, tc, 0, 0])
            for (r, c), v in zip(tl, vals):
                if v != 0:
                    cur[r, c] = v

    # 2. double every column (right to left so sources survive)
    for c in range(wi - 1, -1, -1):
        src = [(r, c) for r in range(hi)]
        tg = [[(r, cc) for r in range(hi)] for cc in (2 * c, 2 * c + 1) if cc != c]
        duplicate_line(src, tg)

    # 3. double every row (bottom to top so sources survive)
    for r in range(hi - 1, -1, -1):
        src = [(r, c) for c in range(W)]
        tg = [[(rr, c) for c in range(W)] for rr in (2 * r, 2 * r + 1) if rr != r]
        duplicate_line(src, tg)

    ops.append(34); sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c59eb873"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c59eb873"
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
                                f"for task c59eb873"
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
                    f"Failed to build a complete episode for task c59eb873 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c59eb873-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
