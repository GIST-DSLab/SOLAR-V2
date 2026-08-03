"""
ARC Task: b8825c91 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 4 is the reserved mask colour; only the canvas colour needs to be episode-fixed
    # (the rule is pure symmetry restoration, independent of the foreground palette).
    bgc = random.choice([c for c in range(10) if c != 4])
    return {"bgc": bgc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int) -> dict:
    cols = remove(4, interval(0, 10, 1))
    hub = max(3, min(15, max_h // 2, max_w // 2))
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = h
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        cands = totuple(bx & dns)
        if len(cands) == 0:
            break
        ch = choice(cands)
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    go = tuple(e for e in gi)
    for alph in (2, 1):
        locidev = unifint(diff_lb, diff_ub, (1, alph * h))
        locjdev = unifint(diff_lb, diff_ub, (1, w))
        loci = alph * h - locidev
        locj = w - locjdev
        loci2 = unifint(diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
        bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
        gi = fill(gi, 4, bd)
        gi, go = rot180(gi), rot180(go)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    # Rule read off I: the picture is mirror-symmetric about both the vertical and the
    # horizontal centre line; rectangular patches of colour 4 hide part of it.  Each
    # patch is rebuilt by copying its mirror partner (still visible in I) and flipping
    # that copy in place.  Nothing is read from O.
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    H, W = I.shape

    G = I.copy()                                  # simulated working grid
    touched = np.zeros((H, W), dtype=bool)        # cells our ops already rewrote
    ops, sels = [], []

    # --- decompose the masked area into whole rectangular 4-patches -----------------
    m = (G == 4)
    rects = []
    while m.any():
        rs, cs = np.nonzero(m)
        r0, c0 = int(rs[0]), int(cs[0])
        c1 = c0
        while c1 + 1 < W and m[r0, c1 + 1]:
            c1 += 1
        r1 = r0
        while r1 + 1 < H and m[r1 + 1, c0:c1 + 1].all():
            r1 += 1
        rects.append((r0, c0, r1, c1))
        m[r0:r1 + 1, c0:c1 + 1] = False

    AXES = ('lr', 'ud', 'rot')

    def source_of(axis, r0, c0, r1, c1):
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if axis == 'lr':
            sr, sc = r0, W - 1 - c1
        elif axis == 'ud':
            sr, sc = H - 1 - r1, c0
        else:
            sr, sc = H - 1 - r1, W - 1 - c1
        if sr < 0 or sc < 0 or sr + bh > H or sc + bw > W:
            return None
        return sr, sc, bh, bw

    remaining = list(rects)
    while remaining:
        progressed = False
        for blk in list(remaining):
            r0, c0, r1, c1 = blk
            # a mirror partner is usable only if it is itself free of 4s
            cands = []
            for axis in AXES:
                s = source_of(axis, r0, c0, r1, c1)
                if s is None:
                    continue
                sr, sc, bh, bw = s
                if (G[sr:sr + bh, sc:sc + bw] == 4).any():
                    continue
                fresh = not touched[sr:sr + bh, sc:sc + bw].any()
                cands.append((0 if fresh else 1, axis, sr, sc, bh, bw))
            if not cands:
                continue
            cands.sort(key=lambda t: t[0])
            fresh_flag, axis, sr, sc, bh, bw = cands[0]

            src = G[sr:sr + bh, sc:sc + bw].copy()

            # copy the mirror partner: from the untouched input when possible,
            # from the working grid when it holds patches we already rebuilt
            ops.append(28 if fresh_flag == 0 else 29)
            sels.append([sr, sc, bh - 1, bw - 1])

            # Paste is transparent to 0 — clear the patch first only if the partner
            # actually contains 0-coloured cells that must land in it
            if (src == 0).any():
                ops.append(0)
                sels.append([r0, c0, bh - 1, bw - 1])

            ops.append(30)
            sels.append([r0, c0, 0, 0])

            cur = src
            if axis in ('lr', 'rot'):
                flipped = np.fliplr(cur)
                if not np.array_equal(flipped, cur):
                    ops.append(26)
                    sels.append([r0, c0, bh - 1, bw - 1])
                cur = flipped
            if axis in ('ud', 'rot'):
                flipped = np.flipud(cur)
                if not np.array_equal(flipped, cur):
                    ops.append(27)
                    sels.append([r0, c0, bh - 1, bw - 1])
                cur = flipped

            G[r0:r1 + 1, c0:c1 + 1] = cur
            touched[r0:r1 + 1, c0:c1 + 1] = True
            remaining.remove(blk)
            progressed = True
        if not progressed:
            break

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
                        f"num_examples+1 ({num_examples + 1}) for task b8825c91"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8825c91"
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
                                f"for task b8825c91"
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
                    f"Failed to build a complete episode for task b8825c91 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8825c91-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
