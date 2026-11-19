"""
ARC Task: ff28f65a (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter, deque

from maker.sel_helpers import sel_of

# Pip layout of the 3x3 board, in lighting order (domino face order).
MPR = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}


def sample_colors(num_examples=None) -> dict:
    # Colours 1 (pips) and 2 (counted objects) are fixed. The background is fixed per episode.
    # The colours of the distractor rectangles are random per rectangle and do not matter to the rule.
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (3, max(3, min(30, max_w))))
    nred = random.randint(1, 5)
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def place(n, colfn):
        succ = 0
        tr = 0
        maxtr = 5 * n
        nonlocal inds
        while tr < maxtr and succ < n:
            tr += 1
            oh = random.randint(1, h // 2 + 1)
            ow = random.randint(1, w // 2 + 1)
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loci, locj = random.choice(cands)
            bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
            if bd.issubset(inds):
                succ += 1
                nb = set()
                for (a, b) in bd:
                    for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nb.add((a + da, b + db))
                inds = (inds - bd) - nb
                col = colfn()
                for (a, b) in bd:
                    gi[a][b] = col
        return succ

    nblue = place(nred, lambda: 2)
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    remcols = [c for c in cols if c != bgc]
    if namt > 0:
        place(namt, lambda: random.choice(remcols))
    go = [[bgc] * 3 for _ in range(3)]
    for k in range(nblue):
        r, c = MPR[k + 1]
        go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # The code-golf entry reads the task the same way the maker does: the 3x3 board is a die
    # face, and its pips light up in a fixed order (TL, TR, centre, BL, BR), one pip per red object.
    # The golf version gets the count from the total red area because the original task's
    # squares all have the same size. Here the red rectangles vary in size, so we count red
    # 4-connected objects. The board stays the same unit, the order stays the same, and the
    # count still sets how many pips light up. So both readings agree, and the maker's route stays.
    import numpy as np
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background = the most common colour other than the red (2) markers
    cnt = Counter(int(x) for x in I.flatten() if int(x) != 2)
    bgc = cnt.most_common(1)[0][0] if cnt else 0

    # count the red objects (4-connected components of colour 2)
    seen = np.zeros((hi, wi), dtype=bool)
    nred = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == 2 and not seen[r, c]:
                nred += 1
                q = deque([(r, c)])
                seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 2:
                            seen[ny, nx] = True
                            q.append((ny, nx))
    nred = min(nred, 5)

    ops, sels = [], []

    # shrink the canvas to the 3x3 board: an all-background 3x3 window of I already IS a blank board
    src = None
    for r in range(hi - 2):
        for c in range(wi - 2):
            if bool((I[r:r + 3, c:c + 3] == bgc).all()):
                src = (r, c)
                break
        if src is not None:
            break

    if src is not None:
        # full 3x3 rectangle window, all background
        ops.append(33); sels.append([src[0], src[1], 2, 2])
    else:
        ops.append(33); sels.append([0, 0, 2, 2])
        # blank the whole 3x3 board to background (full rectangle)
        ops.append(bgc); sels.append([0, 0, 2, 2])

    # light one pip per red object, in pip order
    for k in range(1, nred + 1):
        pr, pc = MPR[k]
        ops.append(1); sels.append(sel_of([(pr, pc)]))

    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
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
                                f"for task ff28f65a"
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
                    f"Failed to build a complete episode for task ff28f65a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff28f65a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
