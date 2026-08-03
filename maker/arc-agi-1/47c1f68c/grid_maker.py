"""
ARC Task: 47c1f68c (RE-ARC) — LLM-generated grid_maker
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
from collections import deque

SCF_VARIANTS = ["identity", "hmirror", "vmirror", "hvmirror"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(1, 10))
    bgc, linc = random.sample(cols, 2)
    objc = random.choice([c for c in cols if c not in (bgc, linc)])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(SCF_VARIANTS):
        examples = [{"scf": v} for v in SCF_VARIANTS]
        examples += [{"scf": random.choice(SCF_VARIANTS)} for _ in range(n_ex - len(SCF_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [{"scf": v} for v in random.sample(SCF_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "linc": linc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, objc, scf=None) -> dict:
    def unifint(bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + int((b - a) * diff_lb)
        hi = a + int((b - a) * diff_ub)
        if hi < lo:
            hi = lo
        return random.randint(lo, hi)

    if scf is None:
        scf = random.choice(SCF_VARIANTS)

    h_ub = max(2, min(14, (min(max_h, 29) - 1) // 2))
    w_ub = max(2, min(14, (min(max_w, 29) - 1) // 2))
    h = unifint((2, h_ub))
    w = unifint((2, w_ub))

    nc = unifint((1, h * w - 1))
    free = {(r, c) for r in range(h) for c in range(w)}
    start = random.choice(sorted(free))
    obj = {start}
    free.discard(start)
    for _ in range(nc - 1):
        cand = set()
        for (r, c) in obj:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    p = (r + dr, c + dc)
                    if p in free:
                        cand.add(p)
        if not cand:
            break
        ch = random.choice(sorted(cand))
        obj.add(ch)
        free.discard(ch)

    quad = np.full((h, w), bgc, dtype=int)
    for (r, c) in obj:
        quad[r, c] = objc
    empty = np.full((h, w), bgc, dtype=int)

    gi = np.full((2 * h + 1, 2 * w + 1), linc, dtype=int)
    gi[0:h, 0:w] = quad
    gi[0:h, w + 1:2 * w + 1] = empty
    gi[h + 1:2 * h + 1, 0:w] = empty
    gi[h + 1:2 * h + 1, w + 1:2 * w + 1] = empty

    base = np.where(quad == objc, linc, quad)
    go = np.zeros((2 * h, 2 * w), dtype=int)
    go[0:h, 0:w] = base
    go[0:h, w:2 * w] = np.fliplr(base)
    go[h:2 * h, 0:w] = np.flipud(base)
    go[h:2 * h, w:2 * w] = np.flipud(np.fliplr(base))

    if scf in ("hmirror", "hvmirror"):
        gi = np.flipud(gi)
        go = np.flipud(go)
    if scf in ("vmirror", "hvmirror"):
        gi = np.fliplr(gi)
        go = np.fliplr(go)

    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape
    # divider row/col sit exactly in the middle, splitting I into four h x w panels
    h, w = (H - 1) // 2, (W - 1) // 2
    linc = int(I[h, 0])

    bgc = None
    src = None
    for (r0, c0) in [(0, 0), (0, w + 1), (h + 1, 0), (h + 1, w + 1)]:
        vals = set(I[r0:r0 + h, c0:c0 + w].flatten().tolist())
        if len(vals) == 1:
            bgc = vals.pop()          # an empty panel reveals the background colour
        else:
            src = (r0, c0)            # the one panel that carries the object
    r0, c0 = src
    panel = I[r0:r0 + h, c0:c0 + w]
    objc = [v for v in set(panel.flatten().tolist()) if v != bgc][0]

    ops, sels = [], []

    # 1. keep only the panel holding the object
    ops.append(33); sels.append([r0, c0, h - 1, w - 1])

    # 2. repaint the object, one connected blob at a time, in the divider colour
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if panel[r, c] != objc or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and panel[ny, nx] == objc:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            ops.append(10 + linc); sels.append([r, c, 0, 0])

    # 3. grow the canvas to the 2x2 mosaic of panels and stash the panel
    ops.append(33); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
    ops.append(29); sels.append([0, 0, h - 1, w - 1])

    # 4. the panel keeps the corner it occupied in I; the other three are its mirrors
    fh0, fv0 = c0 > 0, r0 > 0
    quads = [(0, 0, fh0, fv0), (0, w, not fh0, fv0),
             (h, 0, fh0, not fv0), (h, w, not fh0, not fv0)]
    for qr, qc, fh, fv in quads:
        if (qr, qc) != (0, 0):
            ops.append(30); sels.append([qr, qc, 0, 0])
        if fh:
            ops.append(26); sels.append([qr, qc, h - 1, w - 1])
        if fv:
            ops.append(27); sels.append([qr, qc, h - 1, w - 1])

    ops.append(34); sels.append([0, 0, 2 * h - 1, 2 * w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 47c1f68c"
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
                                f"for task 47c1f68c"
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
                    f"Failed to build a complete episode for task 47c1f68c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"47c1f68c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
