"""
ARC Task: 2bcee788 (RE-ARC) — LLM-generated grid_maker
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

# Each side names where the marker line sits relative to the blob; the
# transform list turns the base layout (marker on the right) into that side.
SIDES = {
    "right":  [["identity"], ["hmirror"]],
    "left":   [["vmirror"], ["vmirror", "hmirror"]],
    "top":    [["rot270"], ["rot270", "vmirror"]],
    "bottom": [["rot90"], ["rot90", "vmirror"]],
}


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    names = list(SIDES)
    if n_ex >= len(names):
        ex = list(names) + [random.choice(names) for _ in range(n_ex - len(names))]
        random.shuffle(ex)
    else:
        ex = random.sample(names, n_ex)
    plan = [{"side": s} for s in ex]
    plan.append(dict(random.choice(plan)))
    return {"bgc": bgc, "sepc": sepc, "objc": objc, "instance_plan": plan}


def _unif(diff_lb, diff_ub, lb, ub):
    if ub <= lb:
        return lb
    d = random.uniform(diff_lb, diff_ub)
    return int(max(lb, min(ub, round(lb + d * (ub - lb)))))


def _apply(g, name):
    if name == "identity":
        return g
    if name == "vmirror":
        return np.fliplr(g)
    if name == "hmirror":
        return np.flipud(g)
    if name == "rot90":
        return np.rot90(g, 3)
    if name == "rot270":
        return np.rot90(g, 1)
    raise ValueError(name)


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sepc, objc, side=None) -> dict:
    if side is None:
        side = random.choice(list(SIDES))
    fns = random.choice(SIDES[side])
    rotates = side in ("top", "bottom")
    # after a rotation the base canvas's height becomes the output's width
    bh, bw = (max_w, max_h) if rotates else (max_h, max_w)
    h = _unif(diff_lb, diff_ub, 2, min(20, bh - 1))
    w = _unif(diff_lb, diff_ub, 2, min(10, (bw - 1) // 2))
    spi = random.randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    allc = {(i, j) for i in range(h) for j in range(w)}
    reminds = allc - {sp}
    numcellsd = _unif(diff_lb, diff_ub, 0, (h * w) // 2)
    numc = random.choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)

    def grow():
        nb = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        nb.add((i + di, j + dj))
        cand = sorted((reminds - shp) & nb)
        shp.add(random.choice(cand))

    for _ in range(numc):
        grow()
    while len({j for _, j in shp}) == 1:
        grow()
    fullh = _unif(diff_lb, diff_ub, h + 1, bh)
    fullw = _unif(diff_lb, diff_ub, 2 * w + 1, bw)
    gi = np.full((fullh, fullw), bgc, dtype=int)
    go = gi.copy()
    loci = random.randint(0, fullh - h)
    locj = random.randint(0, fullw - 2 * w)
    for (i, j) in shp:
        gi[loci + i, locj + j] = objc
        go[loci + i, locj + j] = objc
        go[loci + i, locj + 2 * w - 1 - j] = objc
        if j == w - 1:
            gi[loci + i, locj + w] = sepc
    for fn in fns:
        gi = _apply(gi, fn)
        go = _apply(go, fn)
    go = np.where(go == bgc, 3, go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- roles, read from I: background spans the picture (largest bbox);
    # of the two other colours the main object outnumbers its marker pixels.
    best = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        key = (int((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)), int(len(rs)))
        if best is None or key > best[0]:
            best = (key, int(col))
    bgc = best[1]
    rest = [int(c) for c in np.unique(I) if int(c) != bgc]
    objc = max(rest, key=lambda c: int((I == c).sum()))
    sepc = min(rest, key=lambda c: int((I == c).sum()))
    obj = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    mk = [(int(r), int(c)) for r, c in zip(*np.where(I == sepc))]
    r0 = min(r for r, _ in obj); r1 = max(r for r, _ in obj)
    c0 = min(c for _, c in obj); c1 = max(c for _, c in obj)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    # --- which side of the object the marker pixels touch -> mirror axis
    mcols = {c for _, c in mk}
    mrows = {r for r, _ in mk}
    if mcols == {c1 + 1}:
        side = "right"
    elif mcols == {c0 - 1}:
        side = "left"
    elif mrows == {r0 - 1}:
        side = "top"
    else:
        side = "bottom"
    if side in ("right", "left"):
        flip_op = 26
        dest_r, dest_c = r0, (c1 + 1 if side == "right" else c0 - ow)
        mirror = [(r, c0 + c1 - c + (ow if side == "right" else -ow)) for r, c in obj]
    else:
        flip_op = 27
        dest_r, dest_c = (r0 - oh if side == "top" else r1 + 1), c0
        mirror = [(r0 + r1 - r + (oh if side == "bottom" else -oh), c) for r, c in obj]

    # --- 1. mirror the main object across the edge the markers sit on,
    #        covering the markers (still in the input's own colours)
    if objc == 0:
        # object drawn in 0: Copy/Paste would carry nothing, so draw its
        # reflection directly; it covers every marker pixel.
        ops.append(0); sels.append(sel_of(mirror))
    else:
        if bgc == 0:
            # a transparent paste would leave the markers under the copy's
            # empty cells: clear the destination rectangle first.
            # bbox intended: the whole destination rectangle.
            ops.append(0); sels.append([dest_r, dest_c, oh - 1, ow - 1])
        # bbox intended: the object's whole bounding rectangle, background included
        ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])        # CopyI
        ops.append(30); sels.append([dest_r, dest_c, 0, 0])          # Paste beside it
        src = I[r0:r1 + 1, c0:c1 + 1]
        flipped = np.fliplr(src) if flip_op == 26 else np.flipud(src)
        if not np.array_equal(src, flipped):
            # bbox intended: reflect the whole pasted rectangle in place
            ops.append(flip_op); sels.append([dest_r, dest_c, oh - 1, ow - 1])

    # --- 2. change the background to green: every cell of the current grid
    #        that still holds the background colour
    G = I.copy()
    for (r, c) in mk:
        G[r, c] = bgc
    for (r, c) in mirror:
        G[r, c] = objc
    bg_now = [(int(r), int(c)) for r, c in zip(*np.where(G == bgc))]
    if bg_now:
        ops.append(3); sels.append(sel_of(bg_now))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2bcee788"
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
                                f"for task 2bcee788"
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
                    f"Failed to build a complete episode for task 2bcee788 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2bcee788-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
