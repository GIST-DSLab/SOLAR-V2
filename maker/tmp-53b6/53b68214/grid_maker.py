"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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

# The rule of this task: the input shows the top rows of a picture that repeats
# down the grid with a constant (dr, dc) step; the answer canvas is always 10
# rows tall and as wide as the input.
H_OUT = 10

VARIANTS = [{"mirror": False}, {"mirror": True}]


# --------------------------------------------------------------------------- #
# episode-level colors / structural plan
# --------------------------------------------------------------------------- #
def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# --------------------------------------------------------------------------- #
# helpers shared by generate() (self-validation) and derive_operations()
# --------------------------------------------------------------------------- #
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    ilb = int(round(a + (b - a) * diff_lb))
    iub = int(round(a + (b - a) * diff_ub))
    if iub < ilb:
        ilb, iub = iub, ilb
    ilb = max(a, min(b, ilb))
    iub = max(a, min(b, iub))
    return random.randint(ilb, iub)


def _bg_color(I):
    return int(Counter(np.asarray(I, dtype=int).flatten().tolist()).most_common(1)[0][0])


def _fg_cells(I, bgc):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    return {(r, c): int(I[r, c]) for r in range(h) for c in range(w) if int(I[r, c]) != bgc}


def _detect_step(cells):
    """Periodicity vector of the motif, measured on the INPUT only.

    Same criterion as the task rule: among all shifts (di, dj) whose overlap
    with the pattern is colour-consistent and non-empty, take the one with the
    biggest overlap (tie-break on di*dj, then di, then dj)."""
    best = None
    for di in range(1, 6):
        for dj in range(-10, 10):
            ov = 0
            ok = True
            for (r, c), col in cells.items():
                t = (r + di, c + dj)
                if t in cells:
                    ov += 1
                    if cells[t] != col:
                        ok = False
                        break
            if ok and ov > 0:
                key = (ov, di * dj, di, dj)
                if best is None or key > best:
                    best = key
    if best is None:
        return None
    return best[2], best[3]


def _plan(I):
    """(bgc, dr, dc, motif band, h, w) — everything measured from I."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = _bg_color(I)
    cells = _fg_cells(I, bgc)
    if not cells:
        return None
    step = _detect_step(cells)
    if step is None:
        return None
    dr, dc = step
    if dr >= h:
        return None
    band0 = {(r, c): col for (r, c), col in cells.items() if r < dr}
    if not band0:
        return None
    return bgc, dr, dc, band0, h, w


def _band_cells(band0, dr, dc, k, h, w):
    """The k-th repetition of the motif, restricted to the newly added rows."""
    out = {}
    for (r, c), col in band0.items():
        rr, cc = r + k * dr, c + k * dc
        if h <= rr < H_OUT and 0 <= cc < w:
            out[(rr, cc)] = col
    return out


def _render(I):
    p = _plan(I)
    if p is None:
        return None
    bgc, dr, dc, band0, h, w = p
    out = np.full((H_OUT, w), bgc, dtype=int)
    out[:h] = np.asarray(I, dtype=int)
    for k in range(1, H_OUT):
        if k * dr >= H_OUT:
            break
        for (rr, cc), col in _band_cells(band0, dr, dc, k, h, w).items():
            out[rr, cc] = col
    return out


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def generate(diff_lb, diff_ub, max_h, max_w, bgc, mirror=None) -> dict:
    if mirror is None:
        mirror = random.choice([True, False])
    remcols = [c for c in range(10) if c != bgc]
    w_ub = max(8, min(30, int(max_w)))
    h_ub = max(2, min(6, int(max_h)))

    while True:
        h = _unifint(diff_lb, diff_ub, (2, h_ub))
        w = _unifint(diff_lb, diff_ub, (8, w_ub))
        ncols = _unifint(diff_lb, diff_ub, (1, 9))
        ccols = random.sample(remcols, ncols)
        oh = _unifint(diff_lb, diff_ub, (1, max(1, h // 2)))
        ow = _unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1)))
        bounds = [(i, j) for i in range(oh) for j in range(ow)]
        ncells = _unifint(diff_lb, diff_ub, (1, oh * ow))
        picked = random.sample(bounds, ncells)
        obj = {ij: random.choice(ccols) for ij in picked}
        mi = min(i for i, j in obj)
        mj = min(j for i, j in obj)
        obj = {(i - mi, j - mj): c for (i, j), c in obj.items()}
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1

        locj = random.randint(0, w // 2)
        hoffs = random.randint(0, ow // 2 + 1)
        go = [[bgc] * w for _ in range(H_OUT)]
        for k in range(H_OUT // oh + 1):
            for (i, j), col in obj.items():
                r = i + k * oh
                c = j + locj + k * hoffs
                if 0 <= r < H_OUT and 0 <= c < w:
                    go[r][c] = col

        if len(set(v for row in go[h:] for v in row)) < 2:
            continue

        gi = [row[:] for row in go[:h]]
        if mirror:
            gi = [row[::-1] for row in gi]
            go = [row[::-1] for row in go]

        # keep only instances whose continuation is unambiguously readable
        # from the input alone
        pred = _render(gi)
        if pred is None or not np.array_equal(pred, np.asarray(go, dtype=int)):
            continue

        return {"input": gi, "output": go}


# --------------------------------------------------------------------------- #
# trajectory
# --------------------------------------------------------------------------- #
def derive_operations(I, O=None, examples=None):
    """Continue the repetition down a 10-row canvas.

    Everything below is read off I and the rule alone — O is never consulted,
    so a different answer grid cannot change a single operation.

    The rule is always performed the same way; the *route* through it is not.
    The choices a person actually faces are each settled by a feature of the
    picture in front of them, so one instance is solved copy-by-copy from the
    bottom and the next colour-by-colour from the top, and the same instance
    always gets the same route:

      * stamp        — copy the first repetition and stamp it down, or draw
                       each repetition by hand (a one-cell motif is never
                       worth a clipboard);
      * bottom_up    — start at the copy nearest the input, or at the last one;
      * base_per_copy— lay the whole background down first, or lay the strip
                       of each copy just before drawing into it;
      * by_colour    — sweep one colour across every copy at once instead of
                       finishing a copy before starting the next;
      * colour_first — inside a copy, take the colours in order, or in the
                       order they appear along the band.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    plan = _plan(I)
    if plan is None or h >= H_OUT:
        return [34], [[0, 0, h - 1, w - 1]]
    bgc, dr, dc, band0, h, w = plan

    ops, sels = [], []
    G = np.zeros((H_OUT, w), dtype=int)     # working grid, kept in step with ARCLE
    G[:h] = I

    # 1. grow the canvas to the 10 rows the rule always produces
    #    (a full rectangle, so the bbox selection is exactly the region meant)
    ops.append(33)
    sels.append([0, 0, H_OUT - 1, w - 1])

    # ---- the route knobs, each read off a different feature of the picture --
    colours = sorted(set(band0.values()))
    ncells = len(band0)
    cmin = min(c for (_, c) in band0)
    cmax = max(c for (_, c) in band0)
    rmax = max(r for (r, _) in band0)

    stamp = ncells >= 2 and (cmin + ncells) % 2 == 0
    bottom_up = (w + dr) % 2 == 1
    by_colour = (not stamp) and len(colours) >= 2 and (cmax + len(colours)) % 2 == 0
    colour_first = (len(colours) + abs(dc) + rmax) % 2 == 0
    base_per_copy = (not by_colour) and (h + abs(dc) + cmax) % 2 == 1

    # ---- primitives, each refusing to emit an operation that shows nothing --
    def paint_rect(r0, r1, c0, c1, colour):
        if r0 > r1 or c0 > c1:
            return
        if (G[r0:r1 + 1, c0:c1 + 1] != colour).any():
            ops.append(int(colour))
            sels.append([r0, c0, r1 - r0, c1 - c0])
            G[r0:r1 + 1, c0:c1 + 1] = colour

    def paint_cells(cells, colour):
        cells = sorted(set(cells))
        if not cells or all(G[r, c] == colour for r, c in cells):
            return
        ops.append(int(colour))
        sels.append(sel_of(cells))
        for r, c in cells:
            G[r, c] = colour

    last_src = [None]

    def stamp_copy(k):
        """Stamp the k-th repetition with one Copy of the first band + Paste."""
        sr0, sr1 = 0, rmax
        sc0, sc1 = cmin, cmax
        d_r, d_c = k * dr, cmin + k * dc
        if d_c < 0:                     # runs off the left edge: copy what fits
            sc0 = cmin - d_c
            d_c = 0
        if sc0 > sc1 or d_r >= H_OUT or d_c >= w:
            return
        src = I[sr0:sr1 + 1, sc0:sc1 + 1]
        after = G.copy()
        for i in range(src.shape[0]):
            for j in range(src.shape[1]):
                v = int(src[i, j])
                rr, cc = d_r + i, d_c + j
                if v and rr < H_OUT and cc < w:
                    after[rr, cc] = v
        # a clipboard is worth nothing where the Paste would carry nothing
        if not np.array_equal(after, G):
            rect = (sr0, sc0, sr1, sc1)
            if rect != last_src[0]:
                ops.append(28)
                sels.append([sr0, sc0, sr1 - sr0, sc1 - sc0])   # full rectangle
                last_src[0] = rect
            ops.append(30)
            sels.append([d_r, d_c, 0, 0])                       # paste origin
            G[:, :] = after
        # a Paste carries nothing where the picture is colour 0, so put those
        # cells down by hand where they land (nothing to emit when there are none)
        paint_cells([(d_r + i, d_c + j)
                     for i in range(src.shape[0]) for j in range(src.shape[1])
                     if int(src[i, j]) == 0
                     and h <= d_r + i < H_OUT and d_c + j < w], 0)

    def draw_copy(k):
        """Draw the k-th repetition by hand, one colour at a time."""
        cells = _band_cells(band0, dr, dc, k, h, w)
        if not cells:
            return
        groups = {}
        for rc, col in cells.items():
            groups.setdefault(col, []).append(rc)
        if colour_first:
            order = sorted(groups)
        else:
            order = sorted(groups, key=lambda col: min(groups[col]))
        for col in order:
            paint_cells(groups[col], col)

    # ---- the copies, as row strips of the answer ---------------------------
    strips = []
    for k in range(1, H_OUT):
        if k * dr >= H_OUT:
            break
        strips.append((k, max(k * dr, h), min(k * dr + dr, H_OUT) - 1))
    if bottom_up:
        strips = strips[::-1]

    if by_colour:
        # one sweep per colour over the whole continuation
        if bgc != 0:
            paint_rect(h, H_OUT - 1, 0, w - 1, bgc)     # the rows just added
        groups = {}
        for k, _r0, _r1 in strips:
            for rc, col in _band_cells(band0, dr, dc, k, h, w).items():
                groups.setdefault(col, []).append(rc)
        if colour_first:
            order = sorted(groups)
        else:
            order = sorted(groups, key=lambda col: min(groups[col]), reverse=bottom_up)
        for col in order:
            paint_cells(groups[col], col)
    else:
        if bgc != 0 and not base_per_copy:
            paint_rect(h, H_OUT - 1, 0, w - 1, bgc)     # the rows just added
        for k, r0, r1 in strips:
            if bgc != 0 and base_per_copy:
                paint_rect(r0, r1, 0, w - 1, bgc)       # this copy's strip
            if stamp:
                stamp_copy(k)
            else:
                draw_copy(k)

    ops.append(34)
    sels.append([0, 0, H_OUT - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
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
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
