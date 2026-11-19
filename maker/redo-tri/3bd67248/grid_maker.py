"""
ARC Task: 3bd67248 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    import random
    # rotation: number of CCW np.rot90 applications applied to the base figure
    # fac_mode: "one" -> upscale factor 1, "multi" -> upscale factor >= 2
    ROTATIONS = [0, 1, 2, 3]
    FAC_MODES = ["one", "multi"]
    VARIANTS = [{"rotation": r, "fac_mode": m} for r in ROTATIONS for m in FAC_MODES]

    cols = [c for c in range(10) if c not in (2, 4)]   # 2 and 4 are reserved output colors
    bgc, linc = random.sample(cols, 2)

    n_ex = num_examples if num_examples else 3

    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
    else:
        rots = random.sample(ROTATIONS, min(n_ex, len(ROTATIONS)))
        while len(rots) < n_ex:
            rots.append(random.choice(ROTATIONS))
        if n_ex >= 2:
            modes = ["one", "multi"] + [random.choice(FAC_MODES) for _ in range(n_ex - 2)]
            random.shuffle(modes)
        else:
            modes = [random.choice(FAC_MODES)]
        examples = [{"rotation": rots[i], "fac_mode": modes[i]} for i in range(n_ex)]

    random.shuffle(examples)
    plan = examples + [dict(random.choice(examples))]   # test variant was demonstrated
    return {"bgc": bgc, "linc": linc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, rotation=None, fac_mode=None) -> dict:
    import random
    import numpy as np

    def unif(lo, hi):
        if hi <= lo:
            return lo
        a = int(lo + (hi - lo) * diff_lb)
        b = int(lo + (hi - lo) * diff_ub)
        a = max(lo, min(hi, a))
        b = max(lo, min(hi, b))
        if b < a:
            a, b = b, a
        return random.randint(a, b)

    if rotation is None:
        rotation = random.choice([0, 1, 2, 3])
    if fac_mode is None:
        fac_mode = random.choice(["one", "multi"])

    swap = rotation in (1, 3)                      # rot90 / rot270 transpose the base shape
    room_r, room_c = (max_w, max_h) if swap else (max_h, max_w)
    room_r = min(room_r, 30)
    room_c = min(room_c, 30)

    fmin = 2 if fac_mode == "multi" else 1
    hub = min(15, room_r // fmin)
    wub = min(15, room_c // fmin)
    if hub < 3 or wub < 3:
        hub = min(15, room_r)
        wub = min(15, room_c)
    hub = max(3, hub)
    wub = max(3, wub)

    h = unif(3, hub)
    w = unif(3, wub)

    facmax = min(30 // max(h, w), room_r // h, room_c // w)
    facmax = max(1, facmax)
    if fac_mode == "one" or facmax < 2:
        fac = 1
    else:
        fac = unif(2, facmax)

    # base picture: linc line down column 0 on a bgc canvas
    gi = [[bgc] * w for _ in range(h)]
    for r in range(h):
        gi[r][0] = linc

    go = [row[:] for row in gi]
    for c in range(1, w):                          # 4-band along the bottom edge, past the line
        go[h - 1][c] = 4
    r, c = h - 2, 1                                # 2-ray shot up-right from beside the corner
    while 0 <= r < h and 0 <= c < w:
        go[r][c] = 2
        r -= 1
        c += 1

    A = np.rot90(np.array(gi, dtype=int), rotation)
    B = np.rot90(np.array(go, dtype=int), rotation)
    ones = np.ones((fac, fac), dtype=int)
    A = np.kron(A, ones)
    B = np.kron(B, ones)

    return {"input": A.tolist(), "output": B.tolist()}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    ops, sels = [], []

    # --- read the input: a bgc canvas with a linc line band on one full edge ---
    cnt = Counter(I.flatten().tolist()).most_common()
    linc = cnt[-1][0]                      # the line covers at most 1/3 of the canvas
    cells = np.argwhere(I == linc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())

    if c0 == 0 and c1 == W - 1 and r0 == 0:
        line_edge, fac = 'top', r1 - r0 + 1
    elif c0 == 0 and c1 == W - 1 and r1 == H - 1:
        line_edge, fac = 'bottom', r1 - r0 + 1
    elif r0 == 0 and r1 == H - 1 and c0 == 0:
        line_edge, fac = 'left', c1 - c0 + 1
    else:
        line_edge, fac = 'right', c1 - c0 + 1

    fac = max(1, fac)                       # block size: the whole figure is drawn in fac x fac blocks
    nR, nC = H // fac, W // fac

    # the 4-band always sits on the edge one quarter-turn from the line (fixed chirality)
    four_edge = {'top': 'left', 'left': 'bottom', 'bottom': 'right', 'right': 'top'}[line_edge]

    edges = (line_edge, four_edge)
    br = 0 if 'top' in edges else nR - 1     # corner block where line and 4-band meet
    bc = 0 if 'left' in edges else nC - 1

    def block_cells(bi, bj):
        return [(r, c)
                for r in range(bi * fac, (bi + 1) * fac)
                for c in range(bj * fac, (bj + 1) * fac)]

    # --- 1) draw the 4-band: the whole perpendicular edge, minus the line's own corner block ---
    if four_edge == 'top':
        fb = [(0, j) for j in range(nC)]
    elif four_edge == 'bottom':
        fb = [(nR - 1, j) for j in range(nC)]
    elif four_edge == 'left':
        fb = [(i, 0) for i in range(nR)]
    else:
        fb = [(i, nC - 1) for i in range(nR)]
    fb = [b for b in fb if b != (br, bc)]
    fb.sort(key=lambda b: abs(b[0] - br) + abs(b[1] - bc))   # outward from the corner

    four_cells = []
    for (bi, bj) in fb:
        four_cells.extend(block_cells(bi, bj))
    if four_cells:
        ops.append(4)
        sels.append(sel_of(four_cells))

    # --- 2) shoot the 2-ray diagonally inward, block by block, starting beside the corner ---
    dr = 1 if br == 0 else -1
    dc = 1 if bc == 0 else -1
    bi, bj = br + dr, bc + dc
    while 0 <= bi < nR and 0 <= bj < nC:
        ops.append(2)
        sels.append(sel_of(block_cells(bi, bj)))
        bi += dr
        bj += dc

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bd67248"
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
                                f"for task 3bd67248"
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
                    f"Failed to build a complete episode for task 3bd67248 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bd67248-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
