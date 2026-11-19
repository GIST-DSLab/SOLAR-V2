"""
ARC Task: 83302e8f (RE-ARC) — LLM-generated grid_maker
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
    cols = [c for c in range(10) if c not in (3, 4)]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return int(min(b, max(a, round(a + d * (b - a)))))


def _comps(G, color, diag=False):
    H, W = G.shape
    seen = np.zeros((H, W), dtype=bool)
    out = []
    for r in range(H):
        for c in range(W):
            if G[r, c] != color or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and G[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(sorted(comp))
    return out


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, linc=None, **kw) -> dict:
    if bgc is None or linc is None:
        cc = sample_colors()
        bgc, linc = cc["bgc"], cc["linc"]
    h = _unifint(diff_lb, diff_ub, (2, 5))
    w = _unifint(diff_lb, diff_ub, (2, 5))
    while max_h // (h + 1) < 3 and h > 2:
        h -= 1
    while max_w // (w + 1) < 3 and w > 2:
        w -= 1
    nh = _unifint(diff_lb, diff_ub, (3, max(3, max_h // (h + 1))))
    nw = _unifint(diff_lb, diff_ub, (3, max(3, max_w // (w + 1))))
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = np.full((fullh, fullw), bgc, dtype=int)
    for i in range(h, fullh, h + 1):
        gi[i, :] = linc
    for j in range(w, fullw, w + 1):
        gi[:, j] = linc
    ofc = {(r, c) for r in range(fullh) for c in range(fullw) if gi[r, c] == linc}
    dots = [p for p in ofc
            if all(q in ofc for q in ((p[0] + 1, p[1]), (p[0] - 1, p[1]),
                                      (p[0], p[1] + 1), (p[0], p[1] - 1)))]
    tmp = gi.copy()
    for (r, c) in dots:
        tmp[r, c] = bgc
    lns = _comps(tmp, linc)
    cands = lns + [[d] for d in dots]
    nbreaks = _unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    for obj in random.sample(cands, nbreaks):
        r, c = random.choice(obj)
        gi[r, c] = bgc
    go = gi.copy()
    go[gi == bgc] = 4
    for comp in _comps(gi, bgc):
        if len(comp) == h * w:
            for (r, c) in comp:
                go[r, c] = 3
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    bgc = int(I[0, 0])  # (0,0) is always canvas, never a line cell

    ops, sels = [], []
    # Colour-at-a-time, whole grid per stroke:
    # 1) every canvas-coloured cell in I -> 4 (one selection: all cells of colour bgc)
    bg_cells = [(r, c) for r in range(I.shape[0]) for c in range(I.shape[1]) if I[r, c] == bgc]
    ops.append(4); sels.append(sel_of(bg_cells))

    # 2) on top of that base, all intact lattice cells at once -> 3
    #    (bgc components of I that are solid rectangles of more than one cell)
    intact = []
    for comp in _comps(I, bgc):
        rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
        area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
        if len(comp) == area and len(comp) > 1:
            intact.extend(comp)
    if intact:
        ops.append(3); sels.append(sel_of(sorted(intact)))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 83302e8f"
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
                                f"for task 83302e8f"
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
                    f"Failed to build a complete episode for task 83302e8f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"83302e8f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
