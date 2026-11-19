"""
ARC Task: ae4f1146 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # The generator leaves out colour 1 because 1 is the colour the rule counts.
    cols = [c for c in range(10) if c != 1]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(max(a, round(a + (b - a) * d)), b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 1]
    if bgc is None:
        bgc = random.choice(cols)
    if fgc is None:
        fgc = random.choice([c for c in cols if c != bgc])
    h = _unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = _unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    dh = _unifint(diff_lb, diff_ub, (2, max(2, h // 3)))
    dw = _unifint(diff_lb, diff_ub, (2, max(2, w // 3)))
    num = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * dh * dw))))
    cards = list(range(dh * dw))
    ccards = sorted(random.sample(cards, min(num, len(cards))))
    inds = [(i, j) for i in range(dh) for j in range(dw)]
    sgs = []
    for card in ccards:
        x = set(random.sample(inds, card))
        sgs.append([(1 if (i, j) in x else fgc, (i, j)) for (i, j) in inds])
    go = [[fgc] * dw for _ in range(dh)]
    for v, (i, j) in sgs[-1]:
        go[i][j] = v
    gi = [[bgc] * w for _ in range(h)]
    inds2 = {(i, j) for i in range(h - dh) for j in range(w - dw)}
    maxtr = 10
    for sg in sgs[::-1]:
        if len(inds2) == 0:
            break
        loc = random.choice(tuple(inds2))
        plcd = [(v, (i + loc[0], j + loc[1])) for v, (i, j) in sg]
        tr = 0
        while (not {p for _, p in plcd}.issubset(inds2)) and tr < maxtr:
            loc = random.choice(tuple(inds2))
            plcd = [(v, (i + loc[0], j + loc[1])) for v, (i, j) in sg]
            tr += 1
        if tr < maxtr:
            r0, c0 = loc
            r1, c1 = r0 + dh - 1, c0 + dw - 1
            outb = {(i, j) for i in range(r0 - 1, r1 + 2) for j in range(c0 - 1, c1 + 2)}
            inds2 = inds2 - outb
            for v, (i, j) in plcd:
                gi[i][j] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Reading: the input holds several same-sized fgc/1 tiles, kept apart by the
    # background. The answer is the tile with the most 1-cells, cut out at its
    # own size. The code-golf entry reads it the same way: "the window with the
    # most 1s". Its fixed 3x3 window is just the size used in the original ARC
    # task. Here tile size varies (dh, dw), so each tile is measured as its own
    # connected object. The two readings agree, so the route is unchanged:
    # one crop to the winning tile, then Submit.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the input colour that is neither 1 nor the tile colour.
    # The tile colour is read from the example outputs.
    bgc = None
    if examples:
        out_cols = set()
        for _, eo in examples:
            out_cols |= set(np.asarray(eo, dtype=int).flatten().tolist())
        cand = [c for c in set(I.flatten().tolist()) if c != 1 and c not in out_cols]
        if len(cand) == 1:
            bgc = cand[0]
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    visited = np.zeros((hi, wi), bool)
    best_cells, best_count = None, -1
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] != bgc and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                cnt = sum(1 for (y, x) in cells if I[y, x] == 1)
                if cnt > best_count:
                    best_count, best_cells = cnt, cells

    rs = [y for y, _ in best_cells]
    cs = [x for _, x in best_cells]
    r0, c0 = min(rs), min(cs)
    h = max(rs) - r0 + 1
    w = max(cs) - c0 + 1

    ops, sels = [], []
    # The winning tile is a solid rectangle (every cell is fgc or 1), so its
    # bbox is exactly the cells we mean. Crop the canvas to it.
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task ae4f1146"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae4f1146"
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
                                f"for task ae4f1146"
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
                    f"Failed to build a complete episode for task ae4f1146 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae4f1146-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
