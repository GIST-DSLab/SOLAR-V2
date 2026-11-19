"""
ARC Task: 5ad4f10b (RE-ARC) — LLM-generated grid_maker
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

try:
    from maker.sel_helpers import sel_of
except Exception:  # pragma: no cover - fallback only
    def sel_of(cells):
        return {"cells": [[int(r), int(c)] for r, c in sorted(cells)]}


def sample_colors(num_examples=None) -> dict:
    # bgc / noisec / objc are the three colours the generator samples; the rule
    # (objc blocks -> noisec cells on a bgc canvas) is colour-role dependent,
    # so all three must stay fixed across the whole episode.
    cols = list(range(10))
    bgc, noisec, objc = random.sample(cols, 3)
    return {"bgc": bgc, "noisec": noisec, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, noisec, objc) -> dict:
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh_ub = max(2, min(6, (max_h - 2) // 2))
    ow_ub = max(2, min(6, (max_w - 2) // 2))
    oh = unifint(diff_lb, diff_ub, (2, oh_ub))
    ow = unifint(diff_lb, diff_ub, (2, ow_ub))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nb, ij)) < 4 for nb in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fac_ub = max(2, min((max_h - 2) // oh, (max_w - 2) // ow))
    fac = unifint(diff_lb, diff_ub, (2, fac_ub))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh + 2, max_h))
    w = unifint(diff_lb, diff_ub, (ow + 2, max_w))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(2, len(cands) // 4)))
    namt = min(namt, len(cands))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """The blocky object shrinks in place onto one corner of its own bounding box,
    repainted in the noise colour, and everything else is cropped away.

    Every route choice below (which corner it collapses onto, whether a bgc base is
    laid first or the strays are cleared by hand, which way the pieces are swept,
    whether a piece is grabbed whole or dragged out as rectangles) is read off the
    INPUT only -- the answer grid is never consulted.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # --- background = dominant colour of the border ring (the object never touches it) ---
    border = I[0, :].tolist() + I[-1, :].tolist() + I[:, 0].tolist() + I[:, -1].tolist()
    bgc = Counter(border).most_common(1)[0][0]
    others = [int(c) for c in np.unique(I).tolist() if int(c) != bgc]

    def scale_of(mask):
        H, W = mask.shape
        for k in range(min(H, W), 0, -1):
            if H % k or W % k:
                continue
            small = mask.reshape(H // k, k, W // k, k).any(axis=(1, 3))
            if np.array_equal(small.repeat(k, 0).repeat(k, 1), mask):
                return k
        return 1

    # --- object colour = the one made of solid upscaled blocks; noise is scattered cells ---
    def block_score(c):
        cs = set(map(tuple, np.argwhere(I == c).tolist()))
        iso = sum(
            1 for (r, cc) in cs
            if not any((r + dr, cc + dc) in cs for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        )
        rs = [r for r, _ in cs]
        cl = [cc for _, cc in cs]
        m = (I[min(rs):max(rs) + 1, min(cl):max(cl) + 1] == c)
        return (1 if iso == 0 else 0, scale_of(m), len(cs))

    objc = max(others, key=block_score)
    rest = [c for c in others if c != objc]
    if rest:
        noisec = rest[0]
    else:
        noisec = bgc
        for pair in (examples or []):
            vals = [int(v) for v in np.unique(np.asarray(pair[1])).tolist() if int(v) != bgc]
            if vals:
                noisec = vals[0]
                break

    # --- the upscaled object, its bounding box and its scale factor ---
    cells = np.argwhere(I == objc)
    r0, c0 = int(cells[:, 0].min()), int(cells[:, 1].min())
    r1, c1 = int(cells[:, 0].max()), int(cells[:, 1].max())
    M = (I[r0:r1 + 1, c0:c1 + 1] == objc)
    fac = scale_of(M)
    P = M[::fac, ::fac]                     # the object at scale 1
    ph, pw = P.shape

    # ---- route choices, all read from the input ----
    # 1. the object collapses toward the corner of its box that faces the near side
    #    of the grid: an object sitting high-left folds up-left, one sitting low-right
    #    folds down-right.
    top = ((r0 + r1) / 2.0) <= (hi - 1) / 2.0
    left = ((c0 + c1) / 2.0) <= (wi - 1) / 2.0
    win_r = r0 if top else r1 - ph + 1
    win_c = c0 if left else c1 - pw + 1
    flip_r = not top
    flip_c = not left

    nnoise = int(np.count_nonzero(I == noisec))
    nobj = int(P.sum())
    draw_mode = nnoise % 3        # 0 = lay a bgc base first, 1 = clear strays then draw, 2 = draw then clear
    by_rect = ((nobj + fac) % 2 == 0)   # grab each piece whole, or drag it out as rectangles
    col_first = (ph > pw) or (ph == pw and nnoise % 2 == 1)   # sweep along the long axis

    def to_scan(r, c):
        i, j = r - win_r, c - win_c
        si = (ph - 1 - i) if flip_r else i
        sj = (pw - 1 - j) if flip_c else j
        return si, sj

    def from_scan(si, sj):
        i = (ph - 1 - si) if flip_r else si
        j = (pw - 1 - sj) if flip_c else sj
        return win_r + i, win_c + j

    def scan_key(rc):
        si, sj = to_scan(*rc)
        return (sj, si) if col_first else (si, sj)

    def components(cellset):
        cellset = set(cellset)
        seen, comps = set(), []
        for cell in sorted(cellset, key=scan_key):
            if cell in seen:
                continue
            q, comp = deque([cell]), set()
            seen.add(cell)
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if nb in cellset and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            comps.append(comp)
        comps.sort(key=lambda s: min(scan_key(x) for x in s))
        return comps

    def rect_pieces(comp):
        rem = set(to_scan(r, c) for r, c in comp)
        out = []
        while rem:
            if col_first:
                sj, si = min((b, a) for a, b in rem)
                h = 1
                while (si + h, sj) in rem:
                    h += 1
                w = 1
                while all((si + k, sj + w) in rem for k in range(h)):
                    w += 1
            else:
                si, sj = min(rem)
                w = 1
                while (si, sj + w) in rem:
                    w += 1
                h = 1
                while all((si + h, sj + k) in rem for k in range(w)):
                    h += 1
            piece = set()
            for a in range(h):
                for b in range(w):
                    rem.discard((si + a, sj + b))
                    piece.add(from_scan(si + a, sj + b))
            out.append(piece)
        return out

    ops, sels = [], []
    G = I.copy()

    def sel_for(cs):
        rs = [r for r, _ in cs]
        cl = [c for _, c in cs]
        rr, cc = min(rs), min(cl)
        h = max(rs) - rr + 1
        w = max(cl) - cc + 1
        if len(cs) == h * w:
            return [rr, cc, h - 1, w - 1]   # intended cells are exactly this full rectangle
        return sel_of(cs)

    def paint(color, cs):
        cs = sorted(cs)
        if not cs:
            return
        if all(G[r, c] == color for r, c in cs):
            return                          # every cell already holds it: the op would do nothing
        ops.append(int(color))
        sels.append(sel_for(cs))
        for r, c in cs:
            G[r, c] = color

    def lay(color, cellset):
        for comp in components(cellset):
            if by_rect:
                for piece in rect_pieces(comp):
                    paint(color, piece)
            else:
                paint(color, comp)

    window = [(win_r + i, win_c + j) for i in range(ph) for j in range(pw)]
    shape_cells = [(win_r + i, win_c + j) for i in range(ph) for j in range(pw) if P[i, j]]
    strays = [(r, c) for (r, c) in window
              if (r, c) not in set(shape_cells) and G[r, c] != bgc]

    if draw_mode == 0:
        paint(bgc, window)          # lay the background down over the whole landing area
        lay(noisec, shape_cells)    # then draw the shrunken object on top of it
    elif draw_mode == 1:
        lay(bgc, strays)            # wipe what is in the way, piece by piece
        lay(noisec, shape_cells)
    else:
        lay(noisec, shape_cells)    # draw the shrunken object first
        lay(bgc, strays)            # then tidy away what is left around it

    # --- keep only the shrunken object (selection is exactly that full rectangle) ---
    ops.append(33)
    sels.append([win_r, win_c, ph - 1, pw - 1])
    ops.append(34)
    sels.append([0, 0, ph - 1, pw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5ad4f10b"
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
                                f"for task 5ad4f10b"
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
                    f"Failed to build a complete episode for task 5ad4f10b "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5ad4f10b-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
