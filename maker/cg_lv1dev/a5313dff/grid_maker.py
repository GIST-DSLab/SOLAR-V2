"""
ARC Task: a5313dff (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kwargs) -> dict:
    cols = [c for c in range(10) if c != 1]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h = min(h, max_h)
    w = min(w, max_w)
    gi = np.full((h, w), bgc, dtype=int)

    def bg_components(g):
        hh, ww = g.shape
        seen = np.zeros((hh, ww), dtype=bool)
        comps = []
        for i in range(hh):
            for j in range(ww):
                if g[i, j] == bgc and not seen[i, j]:
                    stack = [(i, j)]
                    seen[i, j] = True
                    comp = []
                    border = False
                    while stack:
                        r, c = stack.pop()
                        comp.append((r, c))
                        if r == 0 or c == 0 or r == hh - 1 or c == ww - 1:
                            border = True
                        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                            if 0 <= nr < hh and 0 <= nc < ww and g[nr, nc] == bgc and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                    comps.append((comp, border))
        return comps

    def has_enclosed(g):
        return any(not b for _, b in bg_components(g))

    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 20)))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    hard_cap = 2000
    while ((tr < maxtr and succ < noccs) or not has_enclosed(gi)) and tr < hard_cap:
        tr += 1
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        lr = random.randint(-1, h)
        lc = random.randint(-1, w)
        ins = [(lr + r, lc + c) for r in range(1, oh - 1) for c in range(1, ow - 1)]
        ins_in = [(r, c) for r, c in ins if 0 <= r < h and 0 <= c < w]
        if any(gi[r, c] == fgc for r, c in ins_in):
            continue
        succ += 1
        for r in range(oh):
            for c in range(ow):
                if r in (0, oh - 1) or c in (0, ow - 1):
                    rr, cc = lr + r, lc + c
                    if 0 <= rr < h and 0 <= cc < w:
                        gi[rr, cc] = fgc
        if random.choice((True, True, False)):
            n_ins = (oh - 2) * (ow - 2)
            k = random.randint(1, max(1, n_ins // 2))
            ss = random.sample(ins, min(k, len(ins)))
            for r, c in ss:
                if 0 <= r < h and 0 <= c < w:
                    gi[r, c] = fgc
    if not has_enclosed(gi):
        raise ValueError("no enclosed region")

    go = gi.copy()
    for comp, border in bg_components(gi):
        if not border:
            for r, c in comp:
                go[r, c] = 1
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the canvas colour shared by every grid of the episode.
    bgc = None
    if examples:
        cnt = Counter()
        for ex_in, _ in examples:
            cnt.update(np.asarray(ex_in, dtype=int).flatten().tolist())
        cnt.update(I.flatten().tolist())
        for col, _ in cnt.most_common():
            if (I == col).any():
                bgc = col
                break
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    FILL = 1  # the task's fixed "interior" colour

    ops, sels = [], []

    # Reading: treat ALL background as interior first (one region = "every cell of
    # colour bgc", nameable from I), then sweep the outside back in from the border:
    # each background region that reaches the grid edge is the outside, restored
    # with one FloodFill(bgc) seeded on its border cell.
    bg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
    ops.append(FILL)
    sels.append(sel_of(bg_cells))

    seen = np.zeros((hi, wi), dtype=bool)
    border_cells = []
    for c in range(wi):
        border_cells.append((0, c))
    for r in range(1, hi):
        border_cells.append((r, wi - 1))
    for c in range(wi - 2, -1, -1):
        border_cells.append((hi - 1, c))
    for r in range(hi - 2, 0, -1):
        border_cells.append((r, 0))

    for (sr, sc) in border_cells:
        if I[sr, sc] != bgc or seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < hi and 0 <= nc < wi and I[nr, nc] == bgc and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        # restore this outside region (currently FILL) back to background
        ops.append(10 + bgc)
        sels.append([sr, sc, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task a5313dff"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a5313dff"
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
                                f"for task a5313dff"
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
                    f"Failed to build a complete episode for task a5313dff "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a5313dff-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
