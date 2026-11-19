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
from dsl import *
from utils import *
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Colour 3 is reserved: it is the output background.
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    return {"bgc": bgc, "sepc": sepc, "objc": objc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sepc=None, objc=None) -> dict:
    if bgc is None or sepc is None or objc is None:
        cols = [c for c in range(10) if c != 3]
        bgc, sepc, objc = random.sample(cols, 3)
    # dmirror / rot90 can swap the dimensions, so both sides have to fit min(max_h, max_w)
    m = min(max_h, max_w, 30)
    h = unifint(diff_lb, diff_ub, (2, max(2, min(20, m - 1))))
    w = unifint(diff_lb, diff_ub, (2, max(2, min(10, (m - 1) // 2))))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(diff_lb, diff_ub, (h + 1, max(h + 1, m)))
    fullw = unifint(diff_lb, diff_ub, (2 * w + 1, max(2 * w + 1, m)))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - h)
    locj = randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, shift(gimini, loc))
    go = paint(fullg, shift(gomini, loc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {"input": gi, "output": go}


def _side_from_cells(obj_rc, sep_rc):
    r0 = min(r for r, _ in obj_rc); r1 = max(r for r, _ in obj_rc)
    c0 = min(c for _, c in obj_rc); c1 = max(c for _, c in obj_rc)
    mc = [c for _, c in sep_rc]
    mr = [r for r, _ in sep_rc]
    if all(c > c1 for c in mc):
        return 'right'
    if all(c < c0 for c in mc):
        return 'left'
    if all(r > r1 for r in mr):
        return 'bottom'
    return 'top'


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- roles, all read from I: the background spans the whole picture
    # (largest bbox area), the blob is the bigger of the other two colours,
    # the 1-wide marker line the smaller.
    best = None
    for col in np.unique(I):
        rs, cs = np.where(I == col)
        key = (int((rs.max() - rs.min() + 1) * (cs.max() - cs.min() + 1)), int(len(rs)))
        if best is None or key > best[0]:
            best = (key, int(col))
    bgc = best[1]
    rest = [int(c) for c in np.unique(I) if int(c) != bgc]
    counts = {c: int((I == c).sum()) for c in rest}
    objc = max(rest, key=lambda c: counts[c])
    sepc = min(rest, key=lambda c: counts[c])

    obj_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == objc))]
    sep_rc = [(int(r), int(c)) for r, c in zip(*np.where(I == sepc))]
    r0 = min(r for r, _ in obj_rc); r1 = max(r for r, _ in obj_rc)
    c0 = min(c for _, c in obj_rc); c1 = max(c for _, c in obj_rc)
    oh, ow = r1 - r0 + 1, c1 - c0 + 1

    # Golf reading: the marker line is where the reversed band of the blob
    # starts. The band is written over the marker, and only at the very end
    # does the background turn into 3.
    side = _side_from_cells(obj_rc, sep_rc)
    if side in ('right', 'left'):
        flip_op = 26
        dr, dc = 0, (ow if side == 'right' else -ow)
        flip_fn = np.fliplr
    else:
        flip_op = 27
        dr, dc = (oh if side == 'bottom' else -oh), 0
        flip_fn = np.flipud
    dest_r, dest_c = r0 + dr, c0 + dc

    G = I.copy()                                # the grid as it stands after each op
    patch = I[r0:r1 + 1, c0:c1 + 1].copy()

    if objc != 0:
        # 1. take the blob's whole band rectangle from the input
        # (bbox selection on purpose: the whole rectangle, background included)
        ops.append(28); sels.append([r0, c0, oh - 1, ow - 1])
        # 2. lay it down starting at the marker line
        ops.append(30); sels.append([dest_r, dest_c, 0, 0])
        landed = [(dest_r + i, dest_c + j)
                  for i in range(oh) for j in range(ow) if patch[i, j] != 0]
        for (r, c) in landed:
            G[r, c] = patch[r - dest_r, c - dest_c]
        # 3. reverse it in place, acting on the cells the paste landed on
        mirrored = flip_fn(patch)
        if not np.array_equal(patch, mirrored):
            ops.append(flip_op); sels.append(sel_of(landed))
            sub = G[dest_r:dest_r + oh, dest_c:dest_c + ow].copy()
            mask = np.zeros((oh, ow), dtype=bool)
            for (r, c) in landed:
                mask[r - dest_r, c - dest_c] = True
            fsub = flip_fn(np.where(mask, sub, 0))
            fmask = flip_fn(mask)
            newsub = sub.copy()
            newsub[mask] = 0 if bgc == 0 else newsub[mask]
            newsub = np.where(mask & ~fmask, sub if bgc != 0 else newsub, newsub)
            newsub = np.where(fmask & (fsub != 0), fsub, newsub)
            G[dest_r:dest_r + oh, dest_c:dest_c + ow] = newsub
    else:
        # The blob is colour 0, which Copy/Paste treat as empty, so paint the
        # reversed blob's own cells. They cover the marker line.
        if side in ('right', 'left'):
            mirror_rc = [(r, c0 + c1 - c + dc) for r, c in obj_rc]
        else:
            mirror_rc = [(r0 + r1 - r + dr, c) for r, c in obj_rc]
        ops.append(0); sels.append(sel_of(mirror_rc))
        for (r, c) in mirror_rc:
            G[r, c] = 0

    # 4. last step: every background-coloured cell of the current grid becomes 3
    bg_now = [(int(r), int(c)) for r, c in zip(*np.where(G == bgc))]
    if bg_now:
        ops.append(3); sels.append(sel_of(bg_now))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
