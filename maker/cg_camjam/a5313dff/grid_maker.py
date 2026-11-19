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
import numpy as np
from collections import Counter
from dsl import *
from utils import unifint
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 1]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, fgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 1]
    if bgc is None or fgc is None:
        bgc, fgc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    gi = canvas(bgc, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    inds = shift(asindices(canvas(-1, (h + 2, w + 2))), (-1, -1))
    indst = totuple(inds)
    while (tr < maxtr and succ < noccs) or len(sfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))) == 0:
        tr += 1
        oh = random.randint(3, 8)
        ow = random.randint(3, 8)
        bx = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        ins = backdrop(inbox(bx))
        loc = random.choice(indst)
        plcdins = shift(ins, loc)
        if len(plcdins & ofcolor(gi, fgc)) == 0:
            succ += 1
            gi = fill(gi, fgc, shift(bx, loc))
            if random.choice((True, True, False)):
                ss = random.sample(totuple(plcdins), random.randint(1, max(1, len(ins) // 2)))
                gi = fill(gi, fgc, ss)
    res = mfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))
    go = fill(gi, 1, res)
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background colour: read from the demonstrations (the colour that turns into 1),
    # falling back to the majority colour of I.
    bgc = None
    if examples:
        votes = Counter()
        for ex_i, ex_o in examples:
            ex_i = np.asarray(ex_i, dtype=int); ex_o = np.asarray(ex_o, dtype=int)
            if ex_i.shape != ex_o.shape:
                continue
            m = (ex_o == 1) & (ex_i != 1)
            votes.update(ex_i[m].tolist())
        if votes:
            cand = votes.most_common(1)[0][0]
            if (I == cand).any():
                bgc = cand
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []
    # Reading (golf): every background cell counts as "inside" unless the outside
    # can reach it from the grid border.
    # Step 1: mark ALL background cells as filled (selection = all cells of one colour).
    bg_cells = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] == bgc]
    ops.append(1); sels.append(sel_of(bg_cells))

    # Step 2: let the outside flow back in. From each border cell still showing 1,
    # flood its connected region back to the background colour.
    cur = I.copy()
    cur[I == bgc] = 1
    border = [(r, c) for r in range(hi) for c in range(wi)
              if r == 0 or c == 0 or r == hi - 1 or c == wi - 1]
    for (r0, c0) in border:
        if cur[r0, c0] != 1:
            continue
        stack = [(r0, c0)]
        cur[r0, c0] = bgc
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < hi and 0 <= nc < wi and cur[nr, nc] == 1:
                    cur[nr, nc] = bgc
                    stack.append((nr, nc))
        ops.append(10 + bgc); sels.append([r0, c0, 0, 0])

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
