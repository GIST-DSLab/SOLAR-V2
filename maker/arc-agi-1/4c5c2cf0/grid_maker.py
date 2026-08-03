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
import numpy as np
from collections import Counter

ROTS = ["identity", "rot90", "rot180", "rot270"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # cc (marker) and objc (pattern) must be nonzero: ARCLE Copy/Paste treats 0 as
    # "nothing", so the mirrored copies must carry only nonzero content.
    nz = [c for c in range(1, 10) if c != bgc]
    cc = random.choice(nz)
    objc = random.choice([c for c in nz if c != cc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        examples = [{"rot": r} for r in ROTS]
        examples += [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(examples)
    else:
        examples = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "cc": cc, "objc": objc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, cc: int, objc: int, rot: str = None) -> dict:
    if rot is None:
        rot = random.choice(ROTS)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 5 and shape(cells) == (3, 3):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule read off I: a 5-cell X-shaped marker (4 diagonal corners + centre) sits in the
    grid; one quadrant next to it holds a pattern block (its inner corner IS the marker's
    inner corner cell).  O = I plus that block mirrored across the marker's centre column,
    centre row, and both.  So: copy the source block once, then paste+flip it into each of
    the three empty quadrants.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- locate the X marker: 5 cells, 3x3 bbox, 4 corners + centre ---
    ci = cj = cc = None
    for c in sorted(set(I.flatten().tolist()) - {bgc}):
        pts = {(int(r), int(k)) for r, k in np.argwhere(I == c)}
        if len(pts) != 5:
            continue
        rs = [p[0] for p in pts]
        ks = [p[1] for p in pts]
        r0, r1, k0, k1 = min(rs), max(rs), min(ks), max(ks)
        if r1 - r0 != 2 or k1 - k0 != 2:
            continue
        mi, mj = r0 + 1, k0 + 1
        if pts == {(r0, k0), (r0, k1), (mi, mj), (r1, k0), (r1, k1)}:
            cc, ci, cj = c, mi, mj
            break

    # --- the pattern colour and the quadrant it lives in ---
    objc = [c for c in sorted(set(I.flatten().tolist())) if c not in (bgc, cc)][0]
    obj = np.argwhere(I == objc)
    orows = obj[:, 0]
    ocols = obj[:, 1]
    sr = -1 if int(orows.max()) < ci else 1
    sc = -1 if int(ocols.max()) < cj else 1

    # source block = pattern bbox extended to the marker corner touching the centre
    r_lo = min(int(orows.min()), ci + sr)
    r_hi = max(int(orows.max()), ci + sr)
    c_lo = min(int(ocols.min()), cj + sc)
    c_hi = max(int(ocols.max()), cj + sc)
    bh, bw = r_hi - r_lo, c_hi - c_lo

    ops, sels = [], []

    # grab the source block once
    ops.append(28); sels.append([r_lo, c_lo, bh, bw])

    # each mirrored quadrant: (dest top-left, flips that realise the mirror)
    targets = [
        (r_lo, 2 * cj - c_hi, [26]),                    # mirror across centre column
        (2 * ci - r_hi, c_lo, [27]),                    # mirror across centre row
        (2 * ci - r_hi, 2 * cj - c_hi, [26, 27]),       # both
    ]
    for dr, dc, flips in targets:
        # bgc==0 -> Paste is transparent there, so the marker cell already sitting in this
        # quadrant would survive and be dragged off-place by the flip: clear it out first.
        if bgc == 0:
            ops.append(0); sels.append([dr, dc, bh, bw])
        ops.append(30); sels.append([dr, dc, 0, 0])
        for f in flips:
            ops.append(f); sels.append([dr, dc, bh, bw])

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
