"""
ARC Task: 4612dd53 (RE-ARC) — LLM-generated grid_maker
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

PLAN = [{"bar_dir": d, "keep_bar": k} for d in ("h", "v") for k in (True, False)]

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, col = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(PLAN):
        ex = [dict(v) for v in PLAN] + [dict(random.choice(PLAN)) for _ in range(n_ex - len(PLAN))]
        random.shuffle(ex)
    else:
        ex = [dict(v) for v in random.sample(PLAN, n_ex)]
    return {"bgc": bgc, "col": col, "instance_plan": ex + [dict(random.choice(ex))]}

def generate(diff_lb, diff_ub, max_h, max_w, bgc, col, bar_dir=None, keep_bar=None) -> dict:
    def unifint(lb, ub):
        a, b = lb, ub
        d = random.uniform(diff_lb, diff_ub)
        return min(b, max(a, int(round(a + d * (b - a)))))
    if bar_dir is None:
        bar_dir = random.choice(("h", "v"))
    if keep_bar is None:
        keep_bar = random.choice((True, False))
    h = unifint(8, max(8, max_h)); w = unifint(8, max(8, max_w))
    ih = unifint(5, h - 1); iw = unifint(5, w - 1)
    loci = random.randint(0, h - ih); locj = random.randint(0, w - iw)
    r1, c1 = loci + ih - 1, locj + iw - 1
    bx = set()
    for c in range(locj, c1 + 1):
        bx.add((loci, c)); bx.add((r1, c))
    for r in range(loci, r1 + 1):
        bx.add((r, locj)); bx.add((r, c1))
    if bar_dir == "h":
        lc = random.randint(loci + 2, loci + ih - 3)
        br = {(lc, c) for c in range(locj + 1, locj + iw - 1)}
    else:
        lc = random.randint(locj + 2, locj + iw - 3)
        br = {(r, lc) for r in range(loci + 1, loci + ih - 1)}
    corners = [(loci, locj), (loci, c1), (r1, locj), (r1, c1)]
    crns = random.sample(corners, 3)
    rembx = sorted(bx - set(crns))
    onbr = random.sample(sorted(br), 2)
    rembr = sorted(br - set(onbr))
    occbx = random.sample(rembx, unifint(0, len(rembx)))
    occbr = random.sample(rembr, unifint(0, len(rembr)))
    c = np.full((h, w), bgc, dtype=int)
    for p in bx | br:
        c[p] = col
    gi = c.copy(); go = c.copy()
    for p in occbx + occbr:
        gi[p] = bgc; go[p] = 2
    if not keep_bar:
        for p in br:
            gi[p] = bgc; go[p] = bgc
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    """
    Reading (shared with the golf entry): the figure is a set of straight grid
    LINES -- every row-line and every column-line of the drawing (the two sides
    of the frame plus the bar, whichever way it runs) is one unit; each line is
    completed end to end with the fill colour wherever it is still background.
    Horizontal lines first (top, bar, bottom), then vertical lines (left, bar, right).
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # fill colour: the colour the examples add that their inputs lack (default 2)
    FILL = 2
    if examples:
        added = set()
        for ei, eo in examples:
            added |= set(np.unique(np.asarray(eo)).tolist()) - set(np.unique(np.asarray(ei)).tolist())
        if len(added) == 1:
            FILL = added.pop()
    # background: the colour of a completely uniform row/column (the frame never spans one)
    bgc = None
    for r in range(hi):
        if len(set(I[r].tolist())) == 1:
            bgc = int(I[r, 0]); break
    if bgc is None:
        for c in range(wi):
            if len(set(I[:, c].tolist())) == 1:
                bgc = int(I[0, c]); break
    if bgc is None:
        bgc = int(np.bincount(I.flatten()).argmax())
    ops, sels = [], []
    pts = np.argwhere(I != bgc)
    if len(pts) == 0:
        return [34], [[0, 0, hi - 1, wi - 1]]
    r0, c0 = int(pts[:, 0].min()), int(pts[:, 1].min())
    r1, c1 = int(pts[:, 0].max()), int(pts[:, 1].max())
    inner = [(int(r), int(c)) for r, c in pts if r0 < r < r1 and c0 < c < c1]
    hrows, vcols = [r0], [c0]
    if inner:
        rows = {r for r, _ in inner}; cols_ = {c for _, c in inner}
        if len(rows) == 1 and (len(cols_) > 1 or (r1 - r0) > (c1 - c0)):
            hrows.append(inner[0][0])
        else:
            vcols.append(inner[0][1])
    hrows.append(r1); vcols.append(c1)
    G = I.copy()
    lines = [[(r, c) for c in range(c0, c1 + 1)] for r in hrows] + \
            [[(r, c) for r in range(r0, r1 + 1)] for c in vcols]
    for line in lines:
        cells = [p for p in line if G[p] == bgc]   # current-grid gaps along this line
        if cells:
            ops.append(FILL); sels.append(sel_of(cells))
            for p in cells:
                G[p] = FILL
    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])  # whole grid: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4612dd53"
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
                                f"for task 4612dd53"
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
                    f"Failed to build a complete episode for task 4612dd53 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4612dd53-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
