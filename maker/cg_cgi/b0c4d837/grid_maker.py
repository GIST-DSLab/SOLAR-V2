"""
ARC Task: b0c4d837 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, boxc, fillc = random.sample(cols, 3)
    return {"bgc": bgc, "boxc": boxc, "fillc": fillc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, boxc=None, fillc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if bgc is None or boxc is None or fillc is None:
        bgc, boxc, fillc = random.sample(range(10), 3)
    H = max(5, min(30, max_h))
    W = max(5, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (5, H))
    w = unifint(diff_lb, diff_ub, (5, W))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = random.randint(0, h - oh)
    locj = random.randint(0, w - ow)
    subg = np.full((oh, ow), boxc, dtype=int)
    subg2 = np.full((oh - 1, ow - 2), fillc, dtype=int)
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh - 2)))
    for j in range(ntofill):
        subg2[j, :] = bgc
    subg[0:oh - 1, 1:ow - 1] = subg2
    gi = np.full((h, w), bgc, dtype=int)
    gi[loci:loci + oh, locj:locj + ow] = subg
    go = [fillc] * ntofill + [bgc] * (9 - ntofill)
    go = [go[:3], go[3:6][::-1], go[6:]]
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    return {"input": gi.tolist(), "output": [list(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    """
    Reading (shared by the maker and the golf entry): the answer is a 3x3 tally
    that repeats the fill colour once per EMPTY interior line of the open box,
    along a snake path (row0 L->R, row1 R->L, row2 L->R), the rest background.

    The golf reading counts the empty lines directly (rows identical to the
    first frame-only row) instead of box_dim - fill_dim - 1.  That is only a
    different way of measuring the same count; the units being replicated (one
    cell per empty line), the order (snake path), and the source colour are the
    same.  So the operations are the same as the maker's: copy one fill cell,
    reframe to 3x3, lay the base, then paste once per counted line.  Here the
    count is measured the golf way, by counting the empty interior lines.
    """
    I = np.asarray(I, dtype=int)

    # colour roles by bounding-box area: background > box frame > fill
    info = []
    for col in np.unique(I):
        cells = np.argwhere(I == col)
        r0, c0 = cells.min(0)
        r1, c1 = cells.max(0)
        info.append((int((r1 - r0 + 1) * (c1 - c0 + 1)), int(col),
                     (int(r0), int(c0), int(r1), int(c1))))
    info.sort(key=lambda t: -t[0])
    bgc = info[0][1]
    boxc, bbox = info[1][1], info[1][2]
    fillc, fbox = info[2][1], info[2][2]

    r0, c0, r1, c1 = bbox
    left_full = all(I[r, c0] == boxc for r in range(r0, r1 + 1))
    right_full = all(I[r, c1] == boxc for r in range(r0, r1 + 1))
    # count the empty interior lines between the two closed parallel sides
    if left_full and right_full:
        lines = [I[r, c0 + 1:c1] for r in range(r0, r1 + 1)]
    else:
        lines = [I[r0 + 1:r1, c] for c in range(c0, c1 + 1)]
    n = sum(1 for ln in lines if len(ln) > 0 and np.all(ln == bgc))
    n = max(0, min(9, n))

    path = []
    for k in range(9):
        r, m = divmod(k, 3)
        c = m if r != 1 else 2 - m
        path.append((r, c))

    # Paste cannot carry colour 0: the colour that is 0 (if any) is the base
    if fillc != 0:
        base_col, rep_col = bgc, fillc
        rep_cells = path[:n]
        src_r, src_c = fbox[0], fbox[1]
    else:
        base_col, rep_col = fillc, bgc
        rep_cells = path[n:]
        bg_cells = np.argwhere(I == bgc)
        src_r, src_c = int(bg_cells[0][0]), int(bg_cells[0][1])

    ops, sels = [], []
    if rep_cells:
        ops.append(28); sels.append(sel_of([(src_r, src_c)]))

    # full 3x3 rectangle -> reframe canvas to the answer grid
    out_cells = [(r, c) for r in range(3) for c in range(3)]
    ops.append(33); sels.append(sel_of(out_cells))

    cur = I[0:3, 0:3].copy()
    cur[cur == 0] = 0
    if not np.all(cur == base_col):
        ops.append(int(base_col)); sels.append(sel_of(out_cells))

    for (r, c) in rep_cells:
        ops.append(30); sels.append(sel_of([(r, c)]))

    ops.append(34); sels.append(sel_of(out_cells))
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
                        f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b0c4d837"
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
                                f"for task b0c4d837"
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
                    f"Failed to build a complete episode for task b0c4d837 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b0c4d837-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
