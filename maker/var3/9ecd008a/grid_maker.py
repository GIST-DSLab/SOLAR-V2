"""
ARC Task: 9ecd008a (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(1, 10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc]
    random.shuffle(palette)
    return {"bgc": bgc, "palette": palette[:8]}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, palette=None) -> dict:
    # final grid is 2h x 2w, so cap h by half of the allowed dims
    hub = min(15, max_h // 2, max_w // 2)
    hub = max(3, hub)
    h = unifint(diff_lb, diff_ub, (3, hub))
    w = h
    if palette is None:
        palette = [c for c in range(1, 10) if c != bgc]
    remcols_all = [c for c in palette if c != bgc]
    if len(remcols_all) == 0:
        remcols_all = [c for c in range(1, 10) if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, min(8, len(remcols_all))))
    remcols = remcols_all[:numcols]

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

    locidev = unifint(diff_lb, diff_ub, (1, 2 * h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2 * h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2 * h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    I is a picture that is symmetric both left-right and up-down, with one
    rectangular window blanked out to 0.  The answer is what that window
    covers, and it is still on the grid: at the window's mirror image across
    whichever axis the window does not straddle (that partner is intact, the
    other one may be eaten by the window itself).

    One rule, three honest ways to carry it out.  Which one is taken is read
    off the input alone -- the size of the window and how far its partner
    sits from it:

      * the window is a big piece of the picture -> mirror the whole picture,
        the missing content lands on the window, then cut the window out;
      * the window is small and its partner sits close by -> copy the partner,
        slot it into the window, mirror it there, then cut the window out;
      * the window is small and its partner lies far off -> cut straight to
        the partner where it lies and mirror that.

    Nothing here is read from O.
    """
    I = np.asarray(I, dtype=int)
    H, W = I.shape

    # the window: the rectangle of 0s (0 is never a picture colour here)
    zr, zc = np.where(I == 0)
    r0, r1 = int(zr.min()), int(zr.max())
    c0, c1 = int(zc.min()), int(zc.max())
    hh, ww = r1 - r0 + 1, c1 - c0 + 1

    # the window's two mirror partners: across the vertical axis (fliplr) and
    # across the horizontal one (flipud)
    cands = []
    for flip_op, sr, sc, ext in ((26, r0, W - 1 - c1, ww),
                                 (27, H - 1 - r1, c0, hh)):
        src = I[sr:sr + hh, sc:sc + ww]
        cands.append({
            "op": flip_op, "sr": sr, "sc": sc, "src": src, "ext": ext,
            "blanks": int((src == 0).sum()),      # 0 -> partner is untouched
            "travel": abs(sr - r0) + abs(sc - c0),
        })
    # an untouched partner first; between two untouched ones take the nearer
    # copy, and if they are equally near the one that mirrors the window's
    # longer side
    cands.sort(key=lambda d: (d["blanks"], d["travel"], -d["ext"]))
    best = cands[0]
    flip_op, sr, sc, src = best["op"], best["sr"], best["sc"], best["src"]

    mirrored = np.fliplr(src) if flip_op == 26 else np.flipud(src)
    needs_flip = not np.array_equal(src, mirrored)   # a self-symmetric patch needs none
    gap = best["travel"] - best["ext"]               # cells lying between window and partner

    ops, sels = [], []
    window  = [r0, c0, hh - 1, ww - 1]   # bbox == exactly the window rectangle
    partner = [sr, sc, hh - 1, ww - 1]   # bbox == exactly the partner rectangle

    if 3 * hh >= H or 3 * ww >= W:
        # the window spans a third of the picture: too big to think of as a
        # patch -- mirror the whole picture, which is what the window's own
        # symmetry says, and read the window off the result
        ops.append(flip_op); sels.append([0, 0, H - 1, W - 1])   # whole grid
        ops.append(33); sels.append(window)
    elif gap <= best["ext"]:
        # the partner all but touches the window: carry it over and repair the
        # picture, then take the window out of the repaired picture
        ops.append(28); sels.append(partner)          # CopyI: the intact partner
        ops.append(30); sels.append([r0, c0, 0, 0])   # paste it into the window
        if needs_flip:
            ops.append(flip_op); sels.append(window)  # mirror it where it now sits
        ops.append(33); sels.append(window)
    else:
        # the partner lies far across the grid: no reason to carry it home --
        # cut straight to it and mirror the cut-out
        ops.append(33); sels.append(partner)
        if needs_flip:
            ops.append(flip_op); sels.append([0, 0, hh - 1, ww - 1])

    # Every route lands its 0s by itself: the whole-grid flip and the in-place
    # flip both keep a zeroed background under the selection, and CropGrid
    # writes into a cleared canvas, so no cell that should read 0 holds
    # anything else.  Nothing to put down with Color(0).
    ops.append(34); sels.append([0, 0, hh - 1, ww - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 9ecd008a"
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
                                f"for task 9ecd008a"
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
                    f"Failed to build a complete episode for task 9ecd008a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"9ecd008a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
