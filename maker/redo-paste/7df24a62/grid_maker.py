"""
ARC Task: 7df24a62 (RE-ARC) — LLM-generated grid_maker
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


# ----------------------------------------------------------------------------- helpers
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        a, b = b, a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(a, min(b, hi))
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


# the eight dihedral orientations a stamped unit can appear in
_NAMES = ["identity", "fliplr", "flipud", "rot180",
          "rot90cw", "rot90ccw", "transpose", "antitranspose"]


def _tf(a, name):
    a = np.asarray(a)
    if name == "identity":
        return np.array(a)
    if name == "fliplr":
        return np.fliplr(a).copy()
    if name == "flipud":
        return np.flipud(a).copy()
    if name == "rot180":
        return np.flipud(np.fliplr(a)).copy()
    cw = np.rot90(a, 3)                      # clockwise
    if name == "rot90cw":
        return cw.copy()
    if name == "rot90ccw":
        return np.rot90(a, 1).copy()
    if name == "transpose":
        return np.fliplr(cw).copy()          # transpose = fliplr o rot90cw
    return np.flipud(cw).copy()              # antitranspose = flipud o rot90cw


def _ops_for(name, square):
    """ARCLE ops that turn a freshly pasted unit into the required orientation."""
    base = {"identity": [], "fliplr": [26], "flipud": [27], "rot180": [26, 27]}
    if name in base:
        return base[name]                    # flips are safe for any shape
    if not square:
        return None                          # rotate needs a square selection
    return {"rot90cw": [25], "rot90ccw": [24],
            "transpose": [25, 26], "antitranspose": [25, 27]}[name]


def _overlap(a, b):
    r1, c1, h1, w1 = a
    r2, c2, h2, w2 = b
    return not (r1 + h1 <= r2 or r2 + h2 <= r1 or c1 + w1 <= c2 or c2 + w2 <= c1)


def _find_occurrences(grid, B, sqc, bgc):
    """Every window of `grid` that holds the unit's noise pattern on bare background,
    in any of the eight orientations. First matching orientation wins (they agree)."""
    grid = np.asarray(grid)
    h, w = grid.shape
    variants = []
    for name in _NAMES:
        Bk = _tf(B, name)
        Tk = np.where(Bk == sqc, bgc, Bk)    # what the bare pattern looks like
        variants.append((name, Bk, Tk))
    occ = []
    for r in range(h):
        for c in range(w):
            for name, Bk, Tk in variants:
                oh, ow = Bk.shape
                if r + oh > h or c + ow > w:
                    continue
                if np.array_equal(grid[r:r + oh, c:c + ow], Tk):
                    occ.append((r, c, name, Bk))
                    break
    return occ


# ----------------------------------------------------------------------------- 1
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, noisec, sqc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "sqc": sqc}


# ----------------------------------------------------------------------------- 2
def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, sqc) -> dict:
    bgc, noisec, sqc = int(bgc), int(noisec), int(sqc)
    hmax = min(32, max(12, int(max_h) + 2))
    wmax = min(32, max(12, int(max_w) + 2))

    for attempt in range(120):
        h = _unifint(diff_lb, diff_ub, (12, hmax))
        w = _unifint(diff_lb, diff_ub, (12, wmax))
        # square framed unit -> every orientation is expressible with ARCLE ops
        s = _unifint(diff_lb, diff_ub, (4, max(4, min(7, h // 3, w // 3))))

        inner = [(i, j) for i in range(s - 2) for j in range(s - 2)]
        obj = {random.choice(inner)}
        while True:
            rs = [p[0] for p in obj]
            cs = [p[1] for p in obj]
            if max(rs) - min(rs) == s - 3 and max(cs) - min(cs) == s - 3:
                break
            remaining = [p for p in inner if p not in obj]
            if not remaining:
                break
            obj.add(random.choice(remaining))

        block = np.full((s, s), sqc, dtype=int)
        for (i, j) in obj:
            block[i + 1, j + 1] = noisec

        gi = np.full((h, w), bgc, dtype=int)
        loci = random.randint(1, h - s - 1)
        locj = random.randint(1, w - s - 1)
        gi[loci:loci + s, locj:locj + s] = block

        forb = {(i, j) for i in range(loci - 1, loci + s + 1)
                for j in range(locj - 1, locj + s + 1)}
        cand = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)
                if (r, c) not in forb]
        if len(cand) < 12:
            continue

        dens = 4 if attempt < 60 else 12
        namt = _unifint(diff_lb, diff_ub, (1, max(1, len(cand) // dens)))
        namt = min(namt, len(cand))
        for (r, c) in random.sample(cand, namt):
            gi[r, c] = noisec

        target = np.where(block == sqc, bgc, block)
        free = set(cand)
        noccs = _unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (s * s * 4))))
        succ, tr = 0, 0
        while succ < noccs and tr < 8 * noccs:
            tr += 1
            name = random.choice(_NAMES)
            t = _tf(target, name)
            r = random.randint(1, h - s - 1)
            c = random.randint(1, w - s - 1)
            cells = {(r + i, c + j) for i in range(s) for j in range(s)}
            if not cells <= free:
                continue
            gi[r:r + s, c:c + s] = t
            for (rr, cc) in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        free.discard((rr + dr, cc + dc))
            succ += 1
        if succ == 0:
            continue

        I = gi[1:h - 1, 1:w - 1].copy()
        br, bc = loci - 1, locj - 1
        B = I[br:br + s, bc:bc + s].copy()

        occ = _find_occurrences(I, B, sqc, bgc)
        if not occ:
            continue
        rects = [(r, c, Bk.shape[0], Bk.shape[1]) for (r, c, nm, Bk) in occ]
        bad = False
        for i in range(len(rects)):
            if _overlap(rects[i], (br, bc, s, s)):
                bad = True
                break
            for j in range(i + 1, len(rects)):
                if _overlap(rects[i], rects[j]):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        O = I.copy()
        for (r, c, nm, Bk) in occ:
            O[r:r + Bk.shape[0], c:c + Bk.shape[1]] = Bk

        return {'input': tuple(tuple(int(v) for v in row) for row in I),
                'output': tuple(tuple(int(v) for v in row) for row in O)}

    raise ValueError('7df24a62: could not build a clean instance')


# ----------------------------------------------------------------------------- 3
def derive_operations(I, O):
    Ia = np.asarray(I, dtype=int)
    Oa = np.asarray(O, dtype=int)
    h, w = Ia.shape
    ops, sels = [], []

    # --- the framed unit: the colour confined to the smallest region -------------
    colors = sorted(set(int(v) for v in Ia.flatten()))
    sqc, best = None, None
    for col in colors:
        pos = np.argwhere(Ia == col)
        bh = int(pos[:, 0].max() - pos[:, 0].min()) + 1
        bw = int(pos[:, 1].max() - pos[:, 1].min()) + 1
        score = (max(bh, bw), bh * bw)
        if best is None or score < best:
            best, sqc = score, col
    pos = np.argwhere(Ia == sqc)
    r0 = int(pos[:, 0].min())
    c0 = int(pos[:, 1].min())
    oh = int(pos[:, 0].max()) - r0 + 1
    ow = int(pos[:, 1].max()) - c0 + 1
    B = Ia[r0:r0 + oh, c0:c0 + ow].copy()

    # background = the most frequent of the remaining colours
    rest = [c for c in colors if c != sqc]
    bgc = max(rest, key=lambda c: int((Ia == c).sum())) if rest else 0

    # --- every bare copy of the unit's pattern, in any orientation ---------------
    occ = _find_occurrences(Ia, B, sqc, bgc)
    occ = [o for o in occ
           if not _overlap((o[0], o[1], o[3].shape[0], o[3].shape[1]),
                           (r0, c0, oh, ow))]

    if occ:
        # clipboard holds the framed unit itself, taken from the input
        ops.append(28)
        sels.append(sel_of([(r0 + i, c0 + j) for i in range(oh) for j in range(ow)]))

    cur = Ia.copy()
    square = (oh == ow)
    for (r, c, name, Bk) in occ:
        bh, bw = Bk.shape
        cells = [(r + i, c + j) for i in range(bh) for j in range(bw)]
        seq = _ops_for(name, square)
        if seq is None:
            # unreachable for this generator (units are square); draw the unit by colour
            for col in sorted(set(int(v) for v in Bk.flatten())):
                tgt = [(r + i, c + j) for i in range(bh) for j in range(bw)
                       if int(Bk[i, j]) == col and int(cur[r + i, c + j]) != col]
                if tgt:
                    ops.append(col)
                    sels.append(sel_of(tgt))
        else:
            win = cur[r:r + bh, c:c + bw]
            if np.any((Bk == 0) & (win != 0)):
                # Paste cannot carry colour-0 cells: lay a 0 base over the whole window
                ops.append(0)
                sels.append(sel_of(cells))
            ops.append(30)
            sels.append(sel_of([(r, c)]))          # stamp the unit here
            for o in seq:
                ops.append(o)
                sels.append(sel_of(cells))         # turn it into the right orientation
        cur[r:r + bh, c:c + bw] = Bk

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])              # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7df24a62"
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
                                f"for task 7df24a62"
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
                    f"Failed to build a complete episode for task 7df24a62 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7df24a62-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
