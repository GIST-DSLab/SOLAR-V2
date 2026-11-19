"""
ARC Task: b527c5c6 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(2, 9)
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (2, 9))
        ccols = sample(remcols, ncols)
    ccols = list(ccols)
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = randint(3, randint(3, (min(h, w)) // 2 - 1))
        d2 = randint(d1 * 2 + 1, randint(d1 * 2 + 1, min(h, w) - 1))
        oh, ow = sample([d1, d2], 2)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(bx)
        if ow < oh:
            lrflag = True
            dcands1 = connect((loci + ow - 1, locj), (loci + oh - 1 - ow + 1, locj))
            dcands2 = shift(dcands1, (0, ow - 1))
        else:
            lrflag = False
            dcands1 = connect((loci, locj + oh - 1), (loci, locj + ow - 1 - oh + 1))
            dcands2 = shift(dcands1, (oh - 1, 0))
        dcands = dcands1 | dcands2
        loc = choice(totuple(dcands))
        sgnflag = -1 if loc in dcands1 else 1
        direc = (sgnflag * (0 if lrflag else 1), sgnflag * (0 if not lrflag else 1))
        ln = shoot(loc, direc)
        shell = set()
        for k in range(min(oh, ow) - 1):
            shell |= power(outbox, k + 1)(ln)
        sqc, dotc = sample(ccols, 2)
        giobj = recolor(sqc, remove(loc, bd)) | {(dotc, loc)}
        goobj = recolor(sqc, (bd | shell) - ln) | recolor(dotc, ln)
        goobj = sfilter(goobj, lambda cij: cij[1] in fullinds)
        goobji = toindices(goobj)
        if goobji.issubset(inds):
            succ += 1
            inds = (inds - goobji) - mapply(dneighbors, bd)
            gi = paint(gi, giobj)
            go = paint(go, goobj)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    # Reading (shared with the code-golf solution): every rectangle carries one odd
    # "dot" on one of its long edges. The strip of that edge centred on the dot,
    # (short side - 1) cells either way, is smeared outward to the grid border:
    # each row (or column) of the strip extends its own edge colour outward.
    # One Color op per smeared line, coloured by the edge cell it continues
    # (Color, not Copy/Paste, because any role colour may be 0).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # Rectangles never touch the border (generator: 1 <= loc <= dim - size - 1),
    # so the corner is always background.
    bgc = int(I[0, 0])
    ops, sels = [], []
    seen = np.zeros((hi, wi), dtype=bool)
    for sr in range(hi):
        for sc in range(wi):
            if I[sr, sc] == bgc or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] and I[nr, nc] != bgc:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cnt = {}
            for r, c in cells:
                cnt[int(I[r, c])] = cnt.get(int(I[r, c]), 0) + 1
            sqc = max(cnt, key=cnt.get)
            dots = [(r, c) for r, c in cells if int(I[r, c]) != sqc]
            if len(dots) != 1:
                continue
            dr, dc = dots[0]
            r0 = min(r for r, _ in cells); r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells); c1 = max(c for _, c in cells)
            oh, ow = r1 - r0 + 1, c1 - c0 + 1
            k = min(oh, ow) - 1
            # order: the dot's own line first, then neighbouring lines outward
            offs = [0]
            for d in range(1, k + 1):
                offs += [-d, d]
            if ow > oh:  # dot on top/bottom edge -> vertical smear
                up = (dr == r0)
                for o in offs:
                    cc = dc + o
                    if up:
                        seg = [(r, cc) for r in range(0, r0)]
                    else:
                        seg = [(r, cc) for r in range(r1 + 1, hi)]
                    ops.append(int(I[dr, cc])); sels.append(sel_of(seg))
            else:  # dot on left/right edge -> horizontal smear
                left = (dc == c0)
                for o in offs:
                    rr = dr + o
                    if left:
                        seg = [(rr, c) for c in range(0, c0)]
                    else:
                        seg = [(rr, c) for c in range(c1 + 1, wi)]
                    ops.append(int(I[rr, dc])); sels.append(sel_of(seg))
    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b527c5c6"
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
                                f"for task b527c5c6"
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
                    f"Failed to build a complete episode for task b527c5c6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b527c5c6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
