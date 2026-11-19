"""
ARC Task: 4c5c2cf0 (RE-ARC) — LLM-generated grid_maker
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

from maker.sel_helpers import sel_of

ROTS = ["identity", "rot90", "rot180", "rot270"]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    return random.randint(a + int((b - a) * diff_lb), a + int((b - a) * diff_ub))


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)                                # colour of the 5-cell X marker
    objc = random.choice([c for c in rem if c != cc])      # colour of the pattern

    # the quadrant the pattern starts in is a discrete variant (the generator rotates
    # the whole scene) -> plan it per instance so every orientation shows up
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cc": cc, "objc": objc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    # rot90/rot270 swap the dimensions -> sample inside the swapped bounds
    if rot in ("rot90", "rot270"):
        bh, bw = max(10, max_w), max(10, max_h)
    else:
        bh, bw = max(10, max_h), max(10, max_w)

    h = _unifint(diff_lb, diff_ub, (10, bh))
    w = _unifint(diff_lb, diff_ub, (10, bw))
    oh = _unifint(diff_lb, diff_ub, (2, max(2, (h - 3) // 2)))
    ow = _unifint(diff_lb, diff_ub, (2, max(2, (w - 3) // 2)))

    locc = (oh - 1, ow - 1)
    reminds = [(i, j) for i in range(oh) for j in range(ow) if (i, j) != locc]

    def _sample_cells():
        n = _unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        n = max(1, min(n, len(reminds)))
        return n, random.sample(reminds, n)

    ncells, cells = _sample_cells()
    guard = 0
    while ncells == 5 and guard < 50:           # never let the pattern itself be a 3x3 X
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        if (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1) != (3, 3):
            break
        ncells, cells = _sample_cells()
        guard += 1

    # the quadrant tile: pattern + the marker cell in its inner corner
    sg = [[bgc] * ow for _ in range(oh)]
    sg[locc[0]][locc[1]] = cc
    for (i, j) in cells:
        sg[i][j] = objc

    # full scene: the tile mirrored into 4 quadrants around a cc centre
    GG = [[bgc] * (2 * ow + 1) for _ in range(2 * oh + 1)]
    for i in range(oh):
        for j in range(ow):
            v = sg[i][j]
            GG[i][j] = v
            GG[i][2 * ow - j] = v
            GG[2 * oh - i][j] = v
            GG[2 * oh - i][2 * ow - j] = v
    GG[oh][ow] = cc

    loci = random.randint(0, h - 2 * oh - 1)
    locj = random.randint(0, w - 2 * ow - 1)

    go = [[bgc] * w for _ in range(h)]
    for i in range(2 * oh + 1):
        for j in range(2 * ow + 1):
            go[loci + i][locj + j] = GG[i][j]

    gi = [[bgc] * w for _ in range(h)]
    for i in range(oh):
        for j in range(ow):
            gi[loci + i][locj + j] = sg[i][j]
    for i in range(h):                          # the whole X marker is visible in the input
        for j in range(w):
            if go[i][j] == cc:
                gi[i][j] = cc

    ai = np.array(gi, dtype=int)
    ao = np.array(go, dtype=int)
    if rot == "rot90":                          # clockwise
        ai, ao = np.rot90(ai, 3), np.rot90(ao, 3)
    elif rot == "rot180":
        ai, ao = np.rot90(ai, 2), np.rot90(ao, 2)
    elif rot == "rot270":                       # counter-clockwise
        ai, ao = np.rot90(ai, 1), np.rot90(ao, 1)

    return {"input": ai.tolist(), "output": ao.tolist()}


def _find_marker(I, bgc):
    """The X: 5 cells of one colour = 3x3 bbox corners + centre (the generator
    forbids the pattern colour from ever taking that shape)."""
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        pts = {(int(r), int(k)) for r, k in np.argwhere(I == col)}
        if len(pts) != 5:
            continue
        rs = [p[0] for p in pts]
        ks = [p[1] for p in pts]
        r0, r1, k0, k1 = min(rs), max(rs), min(ks), max(ks)
        if r1 - r0 != 2 or k1 - k0 != 2:
            continue
        mi, mj = r0 + 1, k0 + 1
        if pts == {(r0, k0), (r0, k1), (mi, mj), (r1, k0), (r1, k1)}:
            return col, mi, mj
    return None


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # background: the colour the generator fills the canvas with before placing anything
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    marker = _find_marker(I, bgc)
    if marker is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    cc, ci, cj = marker

    obj = [(int(r), int(k)) for r, k in np.argwhere((I != bgc) & (I != cc))]
    if not obj:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    objc = int(I[obj[0]])

    orows = [r for r, _ in obj]
    ocols = [c for _, c in obj]
    sr = -1 if max(orows) < ci else 1           # which quadrant the pattern occupies
    sc = -1 if max(ocols) < cj else 1           # (the scene may be rotated any way)

    # source block: the pattern's bbox grown to the marker corner touching the centre
    r_lo = min(min(orows), ci + sr)
    r_hi = max(max(orows), ci + sr)
    c_lo = min(min(ocols), cj + sc)
    c_hi = max(max(ocols), cj + sc)
    bh, bw = r_hi - r_lo, c_hi - c_lo           # bbox offsets

    # (destination top-left, flips that turn the pasted copy into the mirror image).
    # A flip is only listed when that dimension actually spans more than one cell —
    # otherwise the mirror is the identity and the op would change nothing.
    targets = [
        (r_lo, 2 * cj - c_hi, [26] if bw else []),                              # across centre column
        (2 * ci - r_hi, c_lo, [27] if bh else []),                              # across centre row
        (2 * ci - r_hi, 2 * cj - c_hi, ([26] if bw else []) + ([27] if bh else [])),
    ]

    if cc != 0 and objc != 0:
        # every cell of the block is non-zero, so Copy/Paste carries it faithfully.
        # all selections here are whole rectangles on purpose (block incl. background).
        ops.append(28); sels.append([r_lo, c_lo, bh, bw])          # CopyI the block
        for dr, dc, flips in targets:
            if bgc == 0 and flips:
                # a zero background makes Paste transparent there, so this quadrant's
                # own marker cell would survive and be dragged off-place by the flip:
                # lay a clean background over the destination first
                ops.append(0); sels.append([dr, dc, bh, bw])
            ops.append(30); sels.append([dr, dc, 0, 0])            # Paste the block
            for f in flips:
                ops.append(f); sels.append([dr, dc, bh, bw])       # mirror it in place
    else:
        # 0 is "nothing" to Copy/Paste, and here 0 is real content (marker or pattern),
        # so the copies must be drawn instead: one mirror image of the pattern at a time.
        # The X marker maps onto itself, so it is already correct in every quadrant.
        for kind in ("v", "h", "b"):
            cells = []
            for (r, c) in obj:
                nr = r if kind == "v" else 2 * ci - r
                nk = c if kind == "h" else 2 * cj - c
                if 0 <= nr < hi and 0 <= nk < wi:
                    cells.append((nr, nk))
            if cells:
                ops.append(objc); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4c5c2cf0"
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
                                f"for task 4c5c2cf0"
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
                    f"Failed to build a complete episode for task 4c5c2cf0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4c5c2cf0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
