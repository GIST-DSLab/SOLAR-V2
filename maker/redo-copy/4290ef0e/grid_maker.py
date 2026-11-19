"""
ARC Task: 4290ef0e (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """Background + a fixed colour ordering for the concentric rings.

    palette[i] is the colour of ring i (i = 0 is the outermost ring), palette[d]
    is the optional centre dot, so colour roles stay consistent all episode.
    Colour 0 is never used for a ring unless it is the background, so every ring
    object is a genuine (non-zero) ARCLE object that Move can grab.
    """
    cols = list(range(10))
    bgc = random.choice(cols)
    if bgc == 0:
        rest = [c for c in cols if c != 0]
    else:
        rest = [c for c in cols if c != 0 and c != bgc]
    random.shuffle(rest)
    return {"bgc": bgc, "palette": rest}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, palette=None, **kwargs) -> dict:
    if palette is None:
        pal = [c for c in range(10) if c != bgc and c != 0]
        if not pal:
            pal = [c for c in range(1, 10)]
        random.shuffle(pal)
        palette = pal
    palette = list(palette)

    dmax = min(7, len(palette) - 1, max(2, max_h // 4), max(2, max_w // 4))
    dmax = max(2, dmax)

    while True:
        d = unifint(diff_lb, diff_ub, (2, dmax))
        w = d
        lo_h = min(4 * d, max_h)
        lo_w = min(4 * d, max_w)
        fullh = unifint(diff_lb, diff_ub, (lo_h, max(lo_h, max_h)))
        fullw = unifint(diff_lb, diff_ub, (lo_w, max(lo_w, max_w)))

        ccols = palette[:d]
        dotc = palette[d]

        quad = canvas(bgc, (d + 1, d + 1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w - idx + 1)
            quad = fill(quad, c, connect((idx, idx), (idx + linlen - 1, idx)))
            quad = fill(quad, c, connect((idx, idx), (idx, idx + linlen - 1)))
        go = canvas(bgc, (d + 1, 2 * d + 1))
        go = paint(go, asobject(quad))
        go = paint(go, shift(asobject(vmirror(quad)), (0, d)))
        go = vconcat(go, hmirror(go)[1:])
        if choice((True, False)):
            go = fill(go, dotc, {center(asindices(go))})

        objs = sfilter(partition(go), lambda o: color(o) != bgc)
        objs = order(objs, width)
        objs = tuple(reversed(objs))          # outermost (largest) ring first

        gi = canvas(bgc, (fullh, fullw))
        fullinds = asindices(gi)
        inds = asindices(gi)
        anchor = None
        fullsuc = True

        for k, obj in enumerate(objs):
            objn = normalize(obj)
            obji = toindices(objn)
            dd = width(obj)
            dh = max(0, dd // 2 - 1)
            base = sfilter(fullinds, lambda ij: ij[0] <= fullh - dd and ij[1] <= fullw - dd)
            if len(base) == 0:
                fullsuc = False
                break
            if k == 0:
                cands = base                   # outermost ring is always fully on-grid
            else:
                cands = base | shift(base, (-dh, 0)) | shift(base, (0, -dh)) | \
                        shift(base, (dh, 0)) | shift(base, (0, dh))
            cands = totuple(cands)
            succ = False
            loc = None
            for _ in range(50):
                loc = choice(cands)
                cellsat = shift(obji, loc)
                on = cellsat & fullinds
                if not on.issubset(inds):
                    continue
                if k > 0 and any(anchor[0] <= i <= anchor[2] and anchor[1] <= j <= anchor[3]
                                 for i, j in on):
                    continue                   # keep the assembly frame clear
                succ = True
                break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
            if k == 0:
                anchor = (loc[0], loc[1], loc[0] + dd - 1, loc[1] + dd - 1)

        if not fullsuc:
            continue
        break

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Reassemble the scattered concentric rings.

    Everything is measured from I only:
      * each colour forms one (possibly boundary-clipped) fully symmetric ring;
        its true side s = max(observed_h, observed_w) because clipping can only
        eat into ONE of the two dimensions;
      * the missing (clipped) part is recovered by the ring's own h/v mirror
        symmetry;
      * rings are concentric: a ring of side s sits at offset (S-s)/2 inside the
        outermost ring of side S, which is the frame / the answer's size.
    Each ring is then slid (unit Moves) onto its ring position and the frame is
    cropped out.  O is never inspected.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = int(Counter(I.flatten().tolist()).most_common(1)[0][0])
    colors = sorted(set(int(v) for v in I.flatten().tolist()) - {bgc})

    objs = []
    for c in colors:
        ys, xs = np.where(I == c)
        cells = [(int(a), int(b)) for a, b in zip(ys, xs)]
        r_lo = min(a for a, _ in cells); r_hi = max(a for a, _ in cells)
        c_lo = min(b for _, b in cells); c_hi = max(b for _, b in cells)
        oh, ow = r_hi - r_lo + 1, c_hi - c_lo + 1
        s = max(oh, ow)                                   # true side of the ring
        if oh == s:
            r0 = r_lo
        else:                                             # clipped top or bottom
            top_hyp = r_hi - (s - 1)
            r0 = top_hyp if top_hyp < 0 else r_lo
        if ow == s:
            c0 = c_lo
        else:
            left_hyp = c_hi - (s - 1)
            c0 = left_hyp if left_hyp < 0 else c_lo
        rel = set((a - r0, b - c0) for a, b in cells)
        full = set()
        for (a, b) in rel:                                # complete by symmetry
            full.add((a, b))
            full.add((s - 1 - a, b))
            full.add((a, s - 1 - b))
            full.add((s - 1 - a, s - 1 - b))
        objs.append({"color": c, "s": s, "r0": r0, "c0": c0, "rel": rel, "full": full})

    S = max(o["s"] for o in objs)
    anchor = max(objs, key=lambda o: o["s"])
    FR = min(max(anchor["r0"], 0), hi - S)
    FC = min(max(anchor["c0"], 0), wi - S)
    frame_cells = set((r, c) for r in range(FR, FR + S) for c in range(FC, FC + S))

    for o in objs:
        off = (S - o["s"]) // 2
        o["dr"] = FR + off - o["r0"]
        o["dc"] = FC + off - o["c0"]
        o["src"] = [(o["r0"] + a, o["c0"] + b) for (a, b) in sorted(o["rel"])]
        o["dst"] = [(FR + off + a, FC + off + b) for (a, b) in sorted(o["rel"])]
        o["miss"] = [(FR + off + a, FC + off + b) for (a, b) in sorted(o["full"] - o["rel"])]

    ops, sels = [], []
    pending = sorted(objs, key=lambda o: -o["s"])
    while pending:
        pick = None
        for o in pending:
            others = set()
            for p in pending:
                if p is not o:
                    others |= set(p["src"])
            if not ((set(o["dst"]) | set(o["miss"])) & others):
                pick = o
                break
        if pick is None:
            pick = pending[0]
        pending.remove(pick)

        c = pick["color"]
        dr, dc = pick["dr"], pick["dc"]
        cur = list(pick["src"])

        if (dr or dc) and c != 0:
            first = True
            step = 1 if dr > 0 else -1
            for _ in range(abs(dr)):
                ops.append(21 if dr > 0 else 20)
                sels.append(sel_of(cur) if first else sel_of([]))
                first = False
                cur = [(a + step, b) for a, b in cur]
            step = 1 if dc > 0 else -1
            for _ in range(abs(dc)):
                ops.append(22 if dc > 0 else 23)
                sels.append(sel_of(cur) if first else sel_of([]))
                first = False
                cur = [(a, b + step) for a, b in cur]
            if bgc != 0:                       # repair only the truly vacated cells
                hole = sorted((set(pick["src"]) - set(cur)) & frame_cells)
                if hole:
                    ops.append(bgc)
                    sels.append(sel_of(hole))
        elif (dr or dc) and c == 0:
            # colour 0 is invisible to ARCLE's object ops; repaint it instead
            if bgc != 0:
                hole = sorted((set(pick["src"]) - set(pick["dst"])) & frame_cells)
                if hole:
                    ops.append(bgc)
                    sels.append(sel_of(hole))
            ops.append(0)
            sels.append(sel_of(pick["dst"]))

        if pick["miss"]:                       # draw back the boundary-clipped part
            ops.append(c)
            sels.append(sel_of(pick["miss"]))

    # crop to the assembled frame: intended cells are exactly this full rectangle
    ops.append(33); sels.append([FR, FC, S - 1, S - 1])
    ops.append(34); sels.append([0, 0, S - 1, S - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4290ef0e"
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
                                f"for task 4290ef0e"
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
                    f"Failed to build a complete episode for task 4290ef0e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4290ef0e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
