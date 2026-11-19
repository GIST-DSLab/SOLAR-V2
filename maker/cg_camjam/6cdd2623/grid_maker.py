"""
ARC Task: 6cdd2623 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    linc = random.choice(remcols)
    remcols = [c for c in remcols if c != linc]
    nnoisecols = random.randint(1, 7)
    noisecols = random.sample(remcols, nnoisecols)
    return {"bgc": bgc, "linc": linc, "noisecols": noisecols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, noisecols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if linc is None or linc == bgc:
        linc = random.choice([c for c in cols if c != bgc])
    remcols = [c for c in cols if c not in (bgc, linc)]
    if noisecols is None:
        noisecols = random.sample(remcols, unifint(diff_lb, diff_ub, (1, 7)))
    noisecols = [c for c in noisecols if c in remcols] or [random.choice(remcols)]
    nnoisecols = len(noisecols)

    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    ininds = [(r, c) for r in range(1, h - 1) for c in range(1, w)]
    fixinds = random.sample(ininds, nnoisecols)
    for col, (r, c) in zip(noisecols, fixinds):
        gi[r][c] = col
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    fixset = set(fixinds)
    pool = [(r, c) for r in range(h) for c in range(w) if (r, c) not in fixset]
    for (r, c) in random.sample(pool, min(nnoise, len(pool))):
        gi[r][c] = random.choice(remcols)

    aa, bb = random.sample((0, 1), 2)
    nilocs = unifint(diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = unifint(diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = random.sample(range(1, h - 1), nilocs)
    jlocs = random.sample(range(1, w - 1), njlocs)
    for ii in ilocs:
        gi[ii][0] = linc
        gi[ii][w - 1] = linc
        for c in range(w):
            go[ii][c] = linc
    for jj in jlocs:
        gi[0][jj] = linc
        gi[h - 1][jj] = linc
        for r in range(h):
            go[r][jj] = linc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # How this reads the task (following the golf solution):
    #   - The marker colour (linc) is fixed for the whole episode, so it is read from the
    #     demonstrations, not guessed again from this one input.
    #   - A cell's colour depends only on whether its row or its column has a marker pair.
    #     So the lines are drawn first, over the noise, straight from the markers. The
    #     markers are never erased and redrawn.
    #   - Then everything that is neither background nor line colour is noise, and is
    #     cleared in one Color(bgc). That selection is "every cell of a noise colour in
    #     the current grid", which can be read off the grid as it stands at that step.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # The generator paints canvas(bgc) and covers at most 1/3 of it with noise.
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])

    # The line colour is the only non-background colour in a demonstration output.
    linc = None
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            eb = Counter(ei.flatten().tolist()).most_common(1)[0][0]
            oc = set(eo.flatten().tolist()) - {eb}
            if len(oc) == 1:
                linc = int(oc.pop())
                break

    def on_border(r, c):
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    if linc is None or not (I == linc).any():
        # Fallback: the largest colour group whose cells all sit on the border frame.
        groups = {}
        for r in range(h):
            for c in range(w):
                groups.setdefault(int(I[r, c]), []).append((r, c))
        cands = [k for k, v in groups.items()
                 if k != bgc and all(on_border(r, c) for r, c in v)]
        linc = max(cands, key=lambda k: len(groups[k])) if cands else None

    G = I.copy()
    ops, sels = [], []

    if linc is not None:
        rows = [r for r in range(1, h - 1) if I[r, 0] == linc and I[r, w - 1] == linc]
        cols = [c for c in range(1, w - 1) if I[0, c] == linc and I[h - 1, c] == linc]
        # 1. A row whose two border cells face each other with markers becomes a full line.
        #    The selection is that whole row (a true rectangle).
        for r in rows:
            ops.append(int(linc)); sels.append([r, 0, 0, w - 1])
            G[r, :] = linc
        # 2. Same for columns with markers at the top and bottom (a full-column rectangle).
        for c in cols:
            ops.append(int(linc)); sels.append([0, c, h - 1, 0])
            G[:, c] = linc

    # 3. Clear the noise: every cell that is still neither background nor line colour.
    noise = [(r, c) for r in range(h) for c in range(w)
             if G[r, c] != bgc and (linc is None or G[r, c] != linc)]
    if noise:
        ops.append(bgc); sels.append(sel_of(noise))
        for r, c in noise:
            G[r, c] = bgc

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6cdd2623"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6cdd2623"
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
                                f"for task 6cdd2623"
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
                    f"Failed to build a complete episode for task 6cdd2623 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6cdd2623-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
