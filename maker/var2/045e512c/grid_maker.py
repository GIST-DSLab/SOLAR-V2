"""
ARC Task: 045e512c (RE-ARC) — LLM-generated grid_maker
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
    """Only bgc and the master-shape colour are episode-level roles.
    Each direction's colour is READ OFF its indicator in the input, so it may vary
    freely per instance (including being 0 — derive_operations handles that)."""
    cols = list(range(10))
    bgc, objc = random.sample(cols, 2)
    return {"bgc": bgc, "objc": objc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int, objc: int) -> dict:
    cols = interval(0, 10, 1)
    hub = max(11, min(30, max_h))
    wub = max(11, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (11, hub))
    w = unifint(diff_lb, diff_ub, (11, wub))
    while True:
        oh = unifint(diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
        ow = unifint(diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
        bounds = asindices(canvas(-1, (oh, ow)))
        c1 = choice(totuple(connect((0, 0), (oh - 1, 0))))
        c2 = choice(totuple(connect((0, 0), (0, ow - 1))))
        c3 = choice(totuple(connect((oh - 1, ow - 1), (oh - 1, 0))))
        c4 = choice(totuple(connect((oh - 1, ow - 1), (0, ow - 1))))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        ncells = unifint(diff_lb, diff_ub, (0, len(remcands)))
        for k in range(ncells):
            loc = choice(remcands)
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    loci = randint(oh + 1, h - 2 * oh - 1)
    locj = randint(ow + 1, w - 2 * ow - 1)
    loc = (loci, locj)
    remcols = remove(bgc, remove(objc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(recolor(objc, obj), loc)
    gi = paint(gi, obj)
    go = paint(go, obj)
    options = totuple(neighbors((0, 0)))
    ndirs = unifint(diff_lb, diff_ub, (1, 8))
    dirs = sample(options, ndirs)
    dcols = [choice(ccols) for k in range(ndirs)]
    hbars = hfrontier((loci - 2, 0)) | hfrontier((loci + oh + 1, 0))
    vbars = vfrontier((0, locj - 2)) | vfrontier((0, locj + ow + 1))
    bars = hbars | vbars
    ofs = increment((oh, ow))
    for direc, col in zip(dirs, dcols):
        indicatorobj = shift(obj, multiply(direc, increment((oh, ow))))
        indicatorobj = sfilter(indicatorobj, lambda cij: cij[1] in bars)
        if len(indicatorobj) == 0:
            continue
        nindsd = unifint(diff_lb, diff_ub, (0, len(indicatorobj) - 1))
        ninds = len(indicatorobj) - nindsd
        indicatorobj = set(sample(totuple(indicatorobj), ninds))
        if len(indicatorobj) > 0 and len(indicatorobj) < len(obj):
            gi = fill(gi, col, indicatorobj)
            for k in range(1, 10):
                go = fill(go, col, shift(obj, multiply(multiply(k, direc), ofs)))
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """Rule (read entirely from I):
         * the largest 8-connected same-colour blob is the master shape (oh x ow);
         * a step of (oh+1, ow+1) separates consecutive copies;
         * a direction is 'live' when I holds a partial stencil of the master shape
           one step away in that direction; that stencil's colour is the direction's
           colour;
         * every live direction gets FULL copies of the master shape, in that colour,
           at every multiple of the step, out to the edge of the grid.

       The *route* through that rule is chosen from the input itself, so a given
       instance always produces the same trajectory while different instances are
       worked in different orders / different ways:
         * first direction  = the one whose stencil shows the most cells;
         * sweep            = clockwise or anticlockwise, by parity of the master's
                              top-left position;
         * layout           = ring by ring, or one direction at a time, by the
                              master's colour plus the number of live directions;
         * long runs of stamps in one direction are drawn once and copied;
         * a direction whose colour is 0 is always drawn with explicit Color0
           (ARCLE's clipboard cannot carry a 0 cell).
       O is never consulted."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- 8-connected same-colour blobs of non-background cells -----------------
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and I[ny, nx] == col:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
            comps.append(comp)

    if comps:
        best = max(comps, key=len)                 # stencils are strict subsets of it
        rs = [p[0] for p in best]
        cs = [p[1] for p in best]
        top, left = min(rs), min(cs)
        oh = max(rs) - top + 1
        ow = max(cs) - left + 1
        mcol = int(I[best[0][0], best[0][1]])
        sr, sc = oh + 1, ow + 1
        K = max(h // oh, w // ow)

        DIRS_CW = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                   (1, 0), (1, -1), (0, -1), (-1, -1)]

        live = []
        for idx, (dr, dc) in enumerate(DIRS_CW):
            hits = []
            for y, x in best:
                yy, xx = y + dr * sr, x + dc * sc
                if 0 <= yy < h and 0 <= xx < w and I[yy, xx] != bgc:
                    hits.append(int(I[yy, xx]))
            if not hits:
                continue
            col = Counter(hits).most_common(1)[0][0]
            copies = []
            for k in range(1, K + 1):
                cells = [(y + k * dr * sr, x + k * dc * sc) for y, x in best]
                vis = [(y, x) for y, x in cells if 0 <= y < h and 0 <= x < w]
                if not vis:
                    continue
                br, bc = top + k * dr * sr, left + k * dc * sc
                full = (br >= 0 and bc >= 0 and br + oh <= h and bc + ow <= w)
                copies.append((k, vis, full, br, bc))
            if copies:
                live.append({"idx": idx, "col": int(col), "n_ind": len(hits), "copies": copies})

        if live:
            # where to start: the loudest stencil; which way round: the master's corner
            anchor = max(live, key=lambda e: (e["n_ind"], -e["idx"]))
            cw = ((top + left) % 2 == 0)
            if cw:
                seq = sorted(live, key=lambda e: (e["idx"] - anchor["idx"]) % 8)
            else:
                seq = sorted(live, key=lambda e: (anchor["idx"] - e["idx"]) % 8)

            ring_major = ((mcol + len(live)) % 2 == 0)

            if ring_major:
                # ring by ring: every direction's nearest copy, then the next one out
                outward = (mcol % 2 == 0)
                ks = sorted({c[0] for e in seq for c in e["copies"]}, reverse=not outward)
                for k in ks:
                    for e in seq:
                        for (kk, vis, full, br, bc) in e["copies"]:
                            if kk == k:
                                ops.append(e["col"])          # Color<col> (0 included)
                                sels.append(sel_of(vis))
            else:
                # one direction finished before the next is started
                for e in seq:
                    col = e["col"]
                    cps = e["copies"]
                    n_full = sum(1 for c in cps if c[2])
                    # a long run of identical stamps: draw the first, copy the rest.
                    # colour 0 can never travel through the clipboard -> always drawn.
                    stamp = (col != 0 and len(cps) >= 3 and n_full >= 2
                             and cps[0][0] == 1 and cps[0][2])
                    if stamp:
                        _, vis1, _, br1, bc1 = cps[0]
                        ops.append(col)
                        sels.append(sel_of(vis1))
                        # whole rectangular tile, background included -> stamp source
                        ops.append(29)
                        sels.append([br1, bc1, oh - 1, ow - 1])
                        for (kk, vis, full, br, bc) in cps[1:]:
                            if full:
                                ops.append(30)
                                sels.append([br, bc, 0, 0])
                            else:
                                # runs off the edge: the clipboard cannot be placed,
                                # so the sliver that fits is drawn by hand
                                ops.append(col)
                                sels.append(sel_of(vis))
                    else:
                        order = cps if (col % 2 == 0) else list(reversed(cps))
                        for (kk, vis, full, br, bc) in order:
                            ops.append(col)
                            sels.append(sel_of(vis))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 045e512c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 045e512c"
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
                                f"for task 045e512c"
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
                    f"Failed to build a complete episode for task 045e512c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"045e512c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
