"""
ARC Task: 4938f0c2 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- colors ----
ROTS = [0, 1, 2, 3]   # number of CW 90-degree rotations applied to the scene


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    cc = random.choice(rem)                                  # 2x2 marker color
    objc = random.choice([c for c in rem if c != cc])        # pattern color

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS]
        ex += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "cc": cc, "objc": objc, "instance_plan": plan}


# -------------------------------------------------------------- generate ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, cc, objc, rot=None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)

    def ui(bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = int(a + (b - a) * diff_lb)
        hi = int(a + (b - a) * diff_ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    # one row and one column get deleted at the end -> allow h = max_h + 1
    h = ui((10, max_h + 1))
    w = ui((10, max_w + 1))
    oh = ui((2, (h - 3) // 2))
    ow = ui((2, (w - 3) // 2))

    # ---- build the quadrant pattern sg (cc marker at its bottom-right) ----
    reminds = [(r, c) for r in range(oh) for c in range(ow)
               if (r, c) != (oh - 1, ow - 1)]
    cells = []
    for _ in range(100):
        ncells = ui((1, max(1, int((2 / 3) * oh * ow))))
        ncells = max(1, min(ncells, len(reminds)))
        cells = random.sample(reminds, ncells)
        rs = [r for r, _ in cells]
        ks = [c for _, c in cells]
        if not (len(cells) == 4 and max(rs) - min(rs) == 1 and max(ks) - min(ks) == 1):
            break

    sg = [[bgc] * ow for _ in range(oh)]
    for r, c in cells:
        sg[r][c] = objc
    sg[oh - 1][ow - 1] = cc

    # ---- 4-fold mirrored scene with separating bars ----
    G1 = [row[:] for row in sg]
    G2 = [row[::-1] for row in sg]
    G3 = [row[:] for row in sg[::-1]]
    G4 = [row[::-1] for row in sg[::-1]]
    top = [G1[i] + [bgc] + G2[i] for i in range(oh)]
    mid = [[bgc] * ow + [cc] + [bgc] * ow]
    bot = [G3[i] + [bgc] + G4[i] for i in range(oh)]
    GG = top + mid + bot

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
    for r in range(h):
        for c in range(w):
            if go[r][c] == cc:
                gi[r][c] = cc

    # ---- random rotation of the whole scene ----
    for _ in range(rot % 4):
        gi = [list(row) for row in zip(*gi[::-1])]
        go = [list(row) for row in zip(*go[::-1])]

    # ---- delete the center row/col of the cc marker cross ----
    cpos = [(r, c) for r, row in enumerate(gi) for c, v in enumerate(row) if v == cc]
    rs = [r for r, _ in cpos]
    ks = [c for _, c in cpos]
    ccpi = min(rs) + (max(rs) - min(rs) + 1) // 2
    ccpj = min(ks) + (max(ks) - min(ks) + 1) // 2

    def drop(g):
        g = g[:ccpi] + g[ccpi + 1:]
        return [row[:ccpj] + row[ccpj + 1:] for row in g]

    gi = drop(gi)
    go = drop(go)

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


# ------------------------------------------------------- derive_operations ---
def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    full = [0, 0, hi - 1, wi - 1]

    def bg_of(g):
        return int(Counter(np.asarray(g, dtype=int).flatten().tolist()).most_common(1)[0][0])

    def markers(g):
        """colours whose whole cell set is a solid 2x2 block -> {colour: (top, left)}"""
        g = np.asarray(g, dtype=int)
        b = bg_of(g)
        out = {}
        for col in set(g.flatten().tolist()) - {b}:
            pts = np.argwhere(g == col)
            if len(pts) == 4:
                r0, c0 = pts.min(0)
                r1, c1 = pts.max(0)
                if r1 - r0 == 1 and c1 - c0 == 1:
                    out[int(col)] = (int(r0), int(c0))
        return out

    bgc = bg_of(I)
    cand = markers(I)
    if not cand:
        return [34], [full]
    if len(cand) == 1:
        mcol = next(iter(cand))
    else:
        # the marker colour is fixed for the episode: the demonstrations settle it
        votes = Counter()
        for ex in (examples or []):
            ei = ex.get("input") if isinstance(ex, dict) else ex[0]
            if ei is not None:
                for k in markers(ei):
                    votes[k] += 1
        mcol = max(sorted(cand), key=lambda c: votes.get(c, 0))
    br, bc = cand[mcol]

    obj = [(r, c) for r in range(hi) for c in range(wi)
           if int(I[r, c]) not in (bgc, mcol)]
    if not obj:
        return [34], [full]
    objc = int(I[obj[0]])
    r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
    c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)
    # spans symmetric about the marker's vertical / horizontal axis
    sc0 = min(c0, 2 * bc + 1 - c1); sc1 = max(c1, 2 * bc + 1 - c0)
    sr0 = min(r0, 2 * br + 1 - r1); sr1 = max(r1, 2 * br + 1 - r0)

    ops, sels = [], []

    def restore(src_op, rect, cells):
        # Copy/Paste treats 0 as empty; if the picture colour is 0 repaint it instead
        if objc != 0:
            ops.append(src_op); sels.append(rect)
            ops.append(30); sels.append([rect[0], rect[1], 0, 0])
        else:
            ops.append(0); sels.append(sel_of(cells))

    # 1) left/right partner: copy the picture's bbox, mirror the full rectangle
    #    spanning picture and its reflection (only bg + symmetric marker cells
    #    besides the picture), then paste the original back in place.
    B = [r0, c0, r1 - r0, c1 - c0]
    if objc != 0:
        ops.append(28); sels.append(B)                        # CopyI picture bbox
    ops.append(26); sels.append([r0, sc0, r1 - r0, sc1 - sc0])  # FlipH whole span rect
    if objc != 0:
        ops.append(30); sels.append([r0, c0, 0, 0])           # paste original back
    else:
        ops.append(0); sels.append(sel_of(obj))
    # 2) up/down partners: same with the pair of pictures (rectangle row band)
    T = [r0, sc0, r1 - r0, sc1 - sc0]
    pair = obj + [(r, 2 * bc + 1 - c) for r, c in obj]
    if objc != 0:
        ops.append(29); sels.append(T)                        # CopyO the pair band
    ops.append(27); sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])  # FlipV whole rect
    if objc != 0:
        ops.append(30); sels.append([r0, sc0, 0, 0])          # paste pair back
    else:
        ops.append(0); sels.append(sel_of(pair))

    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4938f0c2"
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
                                f"for task 4938f0c2"
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
                    f"Failed to build a complete episode for task 4938f0c2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4938f0c2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
