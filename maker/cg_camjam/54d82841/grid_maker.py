"""
ARC Task: 54d82841 (RE-ARC) — LLM-generated grid_maker
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
    # Color 4 is reserved for the marks. Only the background matters to the rule.
    # Each U-shape's colour can be any other colour, so generate() picks those itself.
    cols = [c for c in range(10) if c != 4]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + d * (b - a)))

    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]

    rot_k = random.choice([0, 1, 2, 3])  # identity, rot90, rot180, rot270
    # Pick the size before rotating, so that the rotated grid still fits max_h x max_w.
    if rot_k in (1, 3):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    h = unifint(diff_lb, diff_ub, (5, min(30, hb)))
    w = unifint(diff_lb, diff_ub, (5, min(30, wb)))
    nshps = unifint(diff_lb, diff_ub, (1, w // 3))

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    locs = list(range(1, w - 1))
    for _ in range(nshps):
        if not locs:
            break
        loc = random.choice(locs)
        locs = [l for l in locs if abs(l - loc) > 2]
        loci = random.randint(1, h - 1)
        col = random.choice(remcols)
        cells = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if dr == 1:
                    continue
                cells.append((loci + dr, loc + dc))
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w:
                gi[r, c] = col
                go[r, c] = col
        go[h - 1, loc] = 4

    # DSL rot90 is clockwise = np.rot90(k=3)
    k_np = {0: 0, 1: 3, 2: 2, 3: 1}[rot_k]
    gi = np.rot90(gi, k_np)
    go = np.rot90(go, k_np)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Orientation: which way the U-mouths open. Every U in the grid opens the same way.
    # A mouth is a background cell with the five U walls of one colour around it.
    TEMPL = {
        "down":  [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)],
        "up":    [(1, -1), (1, 0), (1, 1), (0, -1), (0, 1)],
        "right": [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0)],
        "left":  [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0)],
    }
    votes = {k: 0 for k in TEMPL}
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc:
                continue
            for k, offs in TEMPL.items():
                cells = [(r + dr, c + dc) for dr, dc in offs]
                if not all(0 <= y < h and 0 <= x < w for y, x in cells):
                    continue
                vals = {int(I[y, x]) for y, x in cells}
                if len(vals) == 1 and bgc not in vals:
                    votes[k] += 1
    d = max(votes, key=lambda k: votes[k])

    # The task as line parity: each line that runs toward the facing border crosses
    # at most one U. The U's two side lines hold 2 wall cells each, and its mouth line
    # holds 1 (the cap). So a line with an odd count of non-background cells is a mouth
    # line, and it gets a 4 where it meets the border. Walk along that border in order.
    ops, sels = [], []
    if d in ("down", "up"):
        border_r = h - 1 if d == "down" else 0
        for c in range(w):  # sweep the border row left -> right, one column line at a time
            cnt = int(np.sum(I[:, c] != bgc))
            if cnt % 2 == 1:
                ops.append(4)
                sels.append(sel_of([(border_r, c)]))
    else:
        border_c = w - 1 if d == "right" else 0
        for r in range(h):  # sweep the border column top -> bottom, one row line at a time
            cnt = int(np.sum(I[r, :] != bgc))
            if cnt % 2 == 1:
                ops.append(4)
                sels.append(sel_of([(r, border_c)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 54d82841"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 54d82841"
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
                                f"for task 54d82841"
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
                    f"Failed to build a complete episode for task 54d82841 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"54d82841-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
