"""
ARC Task: 7b6016b9 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (2, 3)]
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    cols = remove(3, remove(2, interval(0, 10, 1)))
    if bgc is None or fgc is None:
        bgc, fgc = sample(cols, 2)
    while True:
        h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
        w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
        numl = unifint(diff_lb, diff_ub, (4, min(h, w)))
        gi = canvas(bgc, (h, w))
        jint = interval(0, w, 1)
        iint = interval(0, h, 1)
        iopts = interval(1, h - 1, 1)
        jopts = interval(1, w - 1, 1)
        numlh = randint(numl // 3, numl // 3 * 2)
        numlw = numl - numlh
        for k in range(numlh):
            if len(iopts) == 0:
                continue
            loci = choice(iopts)
            iopts = remove(loci, iopts)
            iopts = remove(loci + 1, iopts)
            iopts = remove(loci - 1, iopts)
            a, b = sample(jint, 2)
            a = randint(0, a)
            b = randint(b, w - 1)
            gi = fill(gi, fgc, connect((loci, a), (loci, b)))
        for k in range(numlw):
            if len(jopts) == 0:
                continue
            locj = choice(jopts)
            jopts = remove(locj, jopts)
            jopts = remove(locj + 1, jopts)
            jopts = remove(locj - 1, jopts)
            a, b = sample(iint, 2)
            a = randint(0, a)
            b = randint(b, h - 1)
            gi = fill(gi, fgc, connect((a, locj), (b, locj)))
        objs = objects(gi, T, F, F)
        bgobjs = colorfilter(objs, bgc)
        tofill = toindices(mfilter(bgobjs, compose(flip, rbind(bordering, gi))))
        if len(tofill) > 0:
            break
    tofix = mapply(neighbors, tofill) - tofill
    gi = fill(gi, fgc, tofix)
    go = fill(gi, 2, tofill)
    go = replace(go, bgc, 3)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background = dominant colour along the grid's outer frame (as in the verifier).
    border = list(I[0, :]) + list(I[-1, :]) + list(I[:, 0]) + list(I[:, -1])
    bgc = Counter(int(v) for v in border).most_common(1)[0][0]

    ops, sels = [], []
    G = I.copy()

    def flood(seed, new):
        r0, c0 = seed
        old = G[r0, c0]
        stack = [seed]
        G[r0, c0] = new
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < hi and 0 <= nc < wi and G[nr, nc] == old:
                    G[nr, nc] = new
                    stack.append((nr, nc))

    # Step 1: the "outside" — walk the frame and flood every background region
    # that touches it with 3 (open space reached from the edge).
    perim = [(0, c) for c in range(wi)]
    perim += [(r, wi - 1) for r in range(1, hi)]
    perim += [(hi - 1, c) for c in range(wi - 2, -1, -1)]
    perim += [(r, 0) for r in range(hi - 2, 0, -1)]
    for (r, c) in perim:
        if G[r, c] == bgc:
            ops.append(13)
            sels.append([r, c, 0, 0])
            flood((r, c), 3)

    # Step 2: whatever background the outside flood couldn't reach is enclosed:
    # select all cells still of the background colour and paint them 2 together.
    rest = [(r, c) for r in range(hi) for c in range(wi) if G[r, c] == bgc]
    if rest:
        ops.append(2)
        sels.append(sel_of(rest))
        for r, c in rest:
            G[r, c] = 2

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
                        f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7b6016b9"
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
                                f"for task 7b6016b9"
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
                    f"Failed to build a complete episode for task 7b6016b9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7b6016b9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
