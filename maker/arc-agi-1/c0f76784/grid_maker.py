"""
ARC Task: c0f76784 (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque


def sample_colors(num_examples=None) -> dict:
    # Fill colors 6/7/8 are hardcoded by the rule (hole-size -> color); not sampled.
    # Box outline colors are foreground presence only (rule ignores them) -> not fixed.
    # Only background must be fixed. bgc cannot be 6/7/8 (generator excludes them).
    cols = [c for c in range(10) if c not in (6, 7, 8)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    lo_h = min(10, max_h)
    lo_w = min(10, max_w)
    h = random.randint(lo_h, max_h)
    w = random.randint(lo_w, max_w)

    cols = [c for c in range(10) if c not in (6, 7, 8)]
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, numcols)

    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)

    num = random.randint(1, max(1, (h * w) // 20))
    available = {(r, c) for r in range(h) for c in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not available:
            break
        oh = random.choice((3, 4, 5))
        ow = oh
        subs = [(i, j) for (i, j) in available if i < h - oh and j < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        if bd <= available:
            col = random.choice(ccols)
            for (r, c) in bd:
                gi[r, c] = col
                go[r, c] = col
            ccc = oh + 3  # 3->6, 4->7, 5->8
            for r in range(loci + 1, loci + oh - 1):
                for c in range(locj + 1, locj + ow - 1):
                    gi[r, c] = bgc
                    go[r, c] = ccc
            succ += 1
            outbox = {(loci - 1 + a, locj - 1 + b)
                      for a in range(oh + 2) for b in range(ow + 2)}
            available -= bd
            available -= outbox
        tr += 1

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Background = the color the generator paints the whole canvas with (dominant).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # Rule (derived from I): every box outline encloses a square hole of bgc cells.
    # Hole area 1 (1x1) -> 6, area 4 (2x2) -> 7, area 9 (3x3) -> 8.
    size_to_color = {1: 6, 4: 7, 9: 8}

    seen = np.zeros((hi, wi), dtype=bool)
    ops, sels = [], []

    # Find enclosed bgc regions (each box's hole is a bgc component disconnected
    # from the large outer background by its surrounding ring). Emit ONE fill per
    # hole over its whole bbox -> region/object-based, no raster cell painting.
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] != bgc or seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                y, x = q.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and I[ny, nx] == bgc and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            n = len(cells)
            if n not in size_to_color:
                continue
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            r0, c0 = min(ys), min(xs)
            r1, c1 = max(ys), max(xs)
            # confirm a solid square hole (enclosed, matches measured size)
            if (r1 - r0 + 1) * (c1 - c0 + 1) == n and (r1 - r0) == (c1 - c0):
                ops.append(size_to_color[n])
                sels.append([r0, c0, r1 - r0, c1 - c0])

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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
                        f"num_examples+1 ({num_examples + 1}) for task c0f76784"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c0f76784"
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
                                f"for task c0f76784"
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
                    f"Failed to build a complete episode for task c0f76784 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c0f76784-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
