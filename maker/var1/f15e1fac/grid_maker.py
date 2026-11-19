"""
ARC Task: f15e1fac (RE-ARC) — LLM-generated grid_maker
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
# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter

# The 8 orientations of D4 the generator can apply to the finished picture.
TRANSFORMS = ('identity', 'vmirror', 'hmirror', 'rot180',
              'dmirror', 'cmirror', 'rot90', 'rot270')
_TRANSPOSING = ('dmirror', 'cmirror', 'rot90', 'rot270')


def _apply_tf(g, name):
    if name == 'identity':
        return g.copy()
    if name == 'vmirror':
        return np.fliplr(g).copy()
    if name == 'hmirror':
        return np.flipud(g).copy()
    if name == 'rot180':
        return np.rot90(g, 2).copy()
    if name == 'dmirror':
        return np.transpose(g).copy()
    if name == 'cmirror':
        return np.rot90(np.transpose(g), 2).copy()
    if name == 'rot90':
        return np.rot90(g, 3).copy()
    if name == 'rot270':
        return np.rot90(g, 1).copy()
    raise ValueError(name)


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        lo, hi = hi, lo
    return random.randint(max(a, lo), min(b, hi))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 2]
    bgc, linc = random.sample(cols, 2)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(TRANSFORMS):
        examples = [{'transform': t} for t in TRANSFORMS]
        examples += [{'transform': random.choice(TRANSFORMS)}
                     for _ in range(n_ex - len(TRANSFORMS))]
        random.shuffle(examples)
    else:
        examples = [{'transform': t} for t in random.sample(TRANSFORMS, n_ex)]
    plan = examples + [dict(random.choice(examples))]   # test orientation was shown
    return {'bgc': bgc, 'linc': linc, 'instance_plan': plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, transform=None) -> dict:
    if transform is None:
        transform = random.choice(TRANSFORMS)
    # a transposing orientation swaps the two caps, so sample against the swapped
    # bounds and the finished picture still fits inside max_h x max_w
    h_ub, w_ub = (max_w, max_h) if transform in _TRANSPOSING else (max_h, max_w)
    h_ub = max(4, min(30, int(h_ub)))
    w_ub = max(4, min(30, int(w_ub)))
    h = _unifint(diff_lb, diff_ub, (4, h_ub))
    w = _unifint(diff_lb, diff_ub, (4, w_ub))
    nsps = _unifint(diff_lb, diff_ub, (1, (w - 1) // 2))
    ngps = _unifint(diff_lb, diff_ub, (1, (h - 1) // 2))
    spsj = sorted(random.sample(range(1, w - 1), nsps))
    gpsi = sorted(random.sample(range(1, h - 1), ngps))

    gi = np.full((h, w), bgc, dtype=int)
    for jj in spsj:
        gi[0, jj] = linc
    for ii in gpsi:
        gi[ii, 0] = 2

    go = gi.copy()
    ofs = 0
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            cc = jj + ofs
            if 0 <= cc < w:
                go[a:b + 1, cc] = linc
        ofs += 1

    gi = _apply_tf(gi, transform)
    go = _apply_tf(go, transform)
    return {'input': tuple(tuple(int(v) for v in row) for row in gi),
            'output': tuple(tuple(int(v) for v in row) for row in go)}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # the picture is drawn onto the input canvas, so the size never changes:
    # nothing below this line reads O at all
    submit = [0, 0, hi - 1, wi - 1]

    # palette is exactly {bgc, linc, 2}: 2 marks the band dividers, linc the seeds
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    linc = None
    for c in np.unique(I).tolist():
        if c != bgc and c != 2:
            linc = c
    if linc is None:
        return [34], [submit]

    # Canonicalise: 2-markers on the left edge, line seeds on the top edge.
    # The 2-line and the seed-line are perpendicular edge lines, so exactly one
    # dihedral view satisfies both -> orientation is read from I, not assumed.
    idx = np.arange(hi * wi).reshape(hi, wi)
    C = IX = None
    for k in range(4):
        for flip in (False, True):
            g = np.rot90(I, k)
            ix = np.rot90(idx, k)
            if flip:
                g = np.fliplr(g)
                ix = np.fliplr(ix)
            r2, c2 = np.where(g == 2)
            rl, cl = np.where(g == linc)
            if r2.size and rl.size and np.all(c2 == 0) and np.all(rl == 0):
                C, IX = g, ix
                break
        if C is not None:
            break
    if C is None:
        return [34], [submit]

    hc, wc = C.shape
    gps = sorted(set(np.where(C == 2)[0].tolist()))          # marker rows
    sps = sorted(np.where(C[0] == linc)[0].tolist())         # seed columns
    bands = list(zip([0] + gps, [x - 1 for x in gps] + [hc - 1]))

    # The picture: one stripe per seed. Stripe k runs from its seed down to the
    # far edge and steps one column right every time it crosses a 2-marker, so
    # inside band `ofs` it is the run of rows [a, b] in column seed + ofs.
    # segs[ofs] = the strokes of that band, ordered by column.
    segs = []
    for ofs, (a, b) in enumerate(bands):
        band = []
        for si, j in enumerate(sps):
            col = j + ofs
            if col >= wc:                     # stripe has walked off the grid
                continue
            ra, rb = a, b
            while ra <= rb and C[ra, col] == linc:   # seed pixel is already drawn
                ra += 1
            if ra > rb:
                continue
            band.append((col, ra, rb, si))
        segs.append(band)

    # ---- route: how this particular picture is most naturally swept ----------
    # Every choice is read off the input, so one grid always draws the same way
    # and it is the grids that differ, not a die roll.
    #   many stripes / few bands  -> sweep band by band (a row of strokes at a time)
    #   few stripes / many bands  -> follow one whole stripe before the next
    band_major = len(sps) > len(gps)
    # start on the side the stripes crowd toward
    seeds_desc = (sps[0] + sps[-1]) > (wc - 1)
    # start on the side the markers crowd toward
    bands_desc = (gps[0] + gps[-1]) > (hc - 1)

    ops, sels = [], []

    def stroke(col_a, col_b, ra, rb):
        """One Color op over canonical rows ra..rb, columns col_a..col_b."""
        p1 = int(IX[ra, col_a])
        p2 = int(IX[rb, col_b])
        r1, c1 = divmod(p1, wi)
        r2, c2 = divmod(p2, wi)
        ops.append(int(linc))
        # exactly the rectangle of that stroke (a run of the stripe), nothing else
        sels.append([min(r1, r2), min(c1, c2), abs(r1 - r2), abs(c1 - c2)])

    if band_major:
        order = range(len(segs) - 1, -1, -1) if bands_desc else range(len(segs))
        for ofs in order:
            band = segs[ofs]
            if not band:
                continue
            # sweeping sideways, stripes that touch are one wide stroke
            strokes = []
            for col, ra, rb, si in band:               # band is column-sorted
                if strokes and strokes[-1][1] + 1 == col \
                        and strokes[-1][2] == ra and strokes[-1][3] == rb:
                    strokes[-1][1] = col               # widen the stroke
                else:
                    strokes.append([col, col, ra, rb])
            for c0, c1, ra, rb in (strokes[::-1] if seeds_desc else strokes):
                stroke(c0, c1, ra, rb)
    else:
        seed_order = range(len(sps) - 1, -1, -1) if seeds_desc else range(len(sps))
        for si in seed_order:
            pieces = [(ofs, s) for ofs, band in enumerate(segs)
                      for s in band if s[3] == si]
            if bands_desc:
                pieces = pieces[::-1]                  # walk the stripe inward
            for ofs, (col, ra, rb, _si) in pieces:
                stroke(col, col, ra, rb)

    ops.append(34)
    sels.append(submit)
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
                        f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f15e1fac"
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
                                f"for task f15e1fac"
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
                    f"Failed to build a complete episode for task f15e1fac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f15e1fac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
