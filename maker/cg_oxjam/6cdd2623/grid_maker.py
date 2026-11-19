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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    linc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if linc is None or linc == bgc:
        linc = random.choice([c for c in cols if c != bgc])
    h = _unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = _unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    remcols = [c for c in cols if c not in (bgc, linc)]
    nnoisecols = _unifint(diff_lb, diff_ub, (1, 7))
    noisecols = random.sample(remcols, nnoisecols)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    ininds = [(i, j) for i in range(1, h - 1) for j in range(1, w)]
    fixinds = random.sample(ininds, nnoisecols)
    for col, (i, j) in zip(noisecols, fixinds):
        gi[i][j] = col
    nnoise = _unifint(diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    fixset = set(fixinds)
    allinds = [(i, j) for i in range(h) for j in range(w) if (i, j) not in fixset]
    for (i, j) in random.sample(allinds, nnoise):
        gi[i][j] = random.choice(remcols)
    aa, bb = random.sample((0, 1), 2)
    nilocs = _unifint(diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = _unifint(diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = random.sample(range(1, h - 1), nilocs)
    jlocs = random.sample(range(1, w - 1), njlocs)
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
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    # Reading (the maker and the golf entry agree, so nothing changes here):
    # if a row has the marker colour at both ends, the whole row becomes a
    # marker line. The same goes for a column with the marker at top and
    # bottom. Everything else becomes background. The golf entry builds row
    # lines OR column lines on a blank canvas. It uses the same units (whole
    # rows, whole columns) in the same order: clear the noise, draw the rows,
    # then draw the columns.
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    bgc = linc = None
    # The examples fix the colour roles. The background is the most common
    # colour in the example inputs. The marker colour is the only other
    # colour the example outputs use.
    if examples:
        cin, cout = Counter(), set()
        for ex in examples:
            ei = ex["input"] if isinstance(ex, dict) else ex[0]
            eo = ex["output"] if isinstance(ex, dict) else ex[1]
            cin.update(np.asarray(ei, dtype=int).flatten().tolist())
            cout.update(np.asarray(eo, dtype=int).flatten().tolist())
        if cin:
            bgc = int(cin.most_common(1)[0][0])
            others = [c for c in cout if c != bgc]
            if len(others) == 1:
                linc = int(others[0])
    if bgc is None:
        bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    if linc is None:
        pts = defaultdict(list)
        for r in range(h):
            for c in range(w):
                pts[int(I[r, c])].append((r, c))
        onb = lambda p: p[0] in (0, h - 1) or p[1] in (0, w - 1)
        cands = [col for col, ps in pts.items()
                 if col != bgc and all(onb(p) for p in ps)]
        if cands:
            linc = max(cands, key=lambda col: len(pts[col]))

    ops, sels = [], []
    # 1. clear the noise: paint the whole canvas background (the base layer)
    if np.any(I != bgc):
        ops.append(int(bgc)); sels.append([0, 0, h - 1, w - 1])  # full rectangle

    if linc is not None:
        rows = [r for r in range(1, h - 1) if I[r, 0] == linc and I[r, w - 1] == linc]
        cols = [c for c in range(1, w - 1) if I[0, c] == linc and I[h - 1, c] == linc]
        # 2. a row with markers at both ends becomes a full line
        for r in rows:
            ops.append(int(linc)); sels.append([r, 0, 0, w - 1])  # whole row
        # 3. a column with markers at top and bottom becomes a full line
        for c in cols:
            ops.append(int(linc)); sels.append([0, c, h - 1, 0])  # whole column

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
