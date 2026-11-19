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
from collections import Counter, defaultdict
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (8, max(8, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (8, max(8, min(30, max_w))))
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    if linc is None or linc == bgc:
        linc = random.choice(remcols)
    remcols = [c for c in remcols if c != linc]
    nnoisecols = unifint(diff_lb, diff_ub, (1, 7))
    noisecols = random.sample(remcols, nnoisecols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    ininds = [(i + 1, j + 1) for i in range(h - 2) for j in range(w - 1)]
    fixinds = random.sample(ininds, nnoisecols)
    for col, (i, j) in zip(noisecols, fixinds):
        gi[i][j] = col
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    fixset = set(fixinds)
    allinds = [(i, j) for i in range(h) for j in range(w) if (i, j) not in fixset]
    for (i, j) in random.sample(allinds, nnoise):
        gi[i][j] = random.choice(remcols)
    aa, bb = random.sample((0, 1), 2)
    nilocs = unifint(diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = unifint(diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = random.sample(list(range(1, h - 1)), nilocs)
    jlocs = random.sample(list(range(1, w - 1)), njlocs)
    for ii in ilocs:
        gi[ii][0] = linc
        gi[ii][w - 1] = linc
        for j in range(w):
            go[ii][j] = linc
    for jj in jlocs:
        gi[0][jj] = linc
        gi[h - 1][jj] = linc
        for i in range(h):
            go[i][jj] = linc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    # Episode conventions read from the demonstrations: background = dominant colour of
    # the example outputs; line colour = the only other colour in the example outputs.
    bgc, linc = None, None
    if examples:
        cnt = Counter()
        for _, eo in examples:
            cnt.update(np.asarray(eo, dtype=int).flatten().tolist())
        bgc = cnt.most_common(1)[0][0]
        others = [c for c, _ in cnt.most_common() if c != bgc]
        if others:
            linc = others[0]
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    if linc is None:
        pts = defaultdict(list)
        for r in range(h):
            for c in range(w):
                pts[int(I[r, c])].append((r, c))

        def on_border(p):
            return p[0] in (0, h - 1) or p[1] in (0, w - 1)

        cands = [col for col, ps in pts.items()
                 if col != bgc and all(on_border(p) for p in ps)]
        linc = max(cands, key=lambda col: len(pts[col])) if cands else None

    ops, sels = [], []
    G = I.copy()

    if linc is not None:
        # 1. A row is a line wherever the LEFT edge carries the marker colour:
        #    draw the whole row from that marker.
        for r in range(1, h - 1):
            if I[r, 0] == linc:
                ops.append(int(linc)); sels.append(sel_of([(r, c) for c in range(w)]))
                G[r, :] = linc
        # 2. A column is a line wherever the TOP edge carries the marker colour.
        for c in range(1, w - 1):
            if I[0, c] == linc:
                ops.append(int(linc)); sels.append(sel_of([(r, c) for r in range(h)]))
                G[:, c] = linc

    # 3. Everything left that is neither background nor line colour is noise: clear it.
    noise = [(r, c) for r in range(h) for c in range(w)
             if G[r, c] != bgc and (linc is None or G[r, c] != linc)]
    if noise:
        ops.append(int(bgc)); sels.append(sel_of(noise))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
