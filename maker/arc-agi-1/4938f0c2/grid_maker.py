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
import numpy as np
from collections import Counter


def sample_colors(num_examples=None) -> dict:
    # bgc, cc, objc are the three randomly sampled colors of the generator.
    # cc/objc are kept nonzero: 0 is "transparent" for ARCLE Copy/Paste, so real
    # object content must never be 0 for the copy-paste mirroring to be exact.
    cols = list(range(10))
    bgc = random.choice(cols)
    cc = random.choice([c for c in cols if c != 0 and c != bgc])
    objc = random.choice([c for c in cols if c != 0 and c not in (bgc, cc)])
    return {"bgc": bgc, "cc": cc, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, cc: int, objc: int) -> dict:
    h = unifint(diff_lb, diff_ub, (10, max_h + 1))
    w = unifint(diff_lb, diff_ub, (10, max_w + 1))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2 / 3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
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
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    ccpi, ccpj = center(ofcolor(gi, cc))
    gi = gi[:ccpi] + gi[ccpi + 1:]
    gi = tuple(r[:ccpj] + r[ccpj + 1:] for r in gi)
    go = go[:ccpi] + go[ccpi + 1:]
    go = tuple(r[:ccpj] + r[ccpj + 1:] for r in go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    I: one object block whose inner corner touches a 2x2 marker block.
    O: that block mirrored into the 3 remaining quadrants around the marker.
    Plan: CopyI the object quadrant once, Paste it into each mirrored quadrant,
    then FlipH / FlipV in place to mirror it there.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # marker = the color forming exactly one 2x2 block (generator forbids the
    # object from being exactly a 2x2 block of 4 cells)
    cc, r0, c0 = None, 0, 0
    for col in sorted(set(I.flatten().tolist()) - {bgc}):
        pts = np.argwhere(I == col)
        if len(pts) == 4:
            rs = sorted(set(pts[:, 0].tolist()))
            cs = sorted(set(pts[:, 1].tolist()))
            if len(rs) == 2 and len(cs) == 2 and rs[1] == rs[0] + 1 and cs[1] == cs[0] + 1:
                cc, r0, c0 = col, rs[0], cs[0]

    obj = np.argwhere((I != bgc) & (I != cc))
    top = int(obj[:, 0].max()) <= r0
    left = int(obj[:, 1].max()) <= c0

    sa = int(obj[:, 0].min()) if top else r0 + 1
    sb = r0 if top else int(obj[:, 0].max())
    sc = int(obj[:, 1].min()) if left else c0 + 1
    sd = c0 if left else int(obj[:, 1].max())
    h1, w1 = sb - sa + 1, sd - sc + 1

    # CopyI the source quadrant (a true full rectangle, background included)
    ops.append(28); sels.append([sa, sc, h1 - 1, w1 - 1])

    for is_top in (True, False):
        for is_left in (True, False):
            if is_top == top and is_left == left:
                continue
            dr = r0 - h1 + 1 if is_top else r0 + 1
            dc = c0 - w1 + 1 if is_left else c0 + 1
            # when bgc == 0 the clipboard's bgc cells are transparent, so the
            # marker cell already sitting in this quadrant would survive the
            # paste; clear it first
            if bgc == 0:
                mr = r0 if is_top else r0 + 1
                mc = c0 if is_left else c0 + 1
                ops.append(0); sels.append([mr, mc, 0, 0])
            ops.append(30); sels.append([dr, dc, 0, 0])
            rect = [dr, dc, h1 - 1, w1 - 1]  # exact rectangle to mirror
            if is_left != left:
                ops.append(26); sels.append(list(rect))
            if is_top != top:
                ops.append(27); sels.append(list(rect))

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
