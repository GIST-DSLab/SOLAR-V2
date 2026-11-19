"""
ARC Task: 5daaa586 (RE-ARC) — LLM-generated grid_maker
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
    import random
    cols = list(range(10))
    bgc = random.choice(cols)
    rem = [c for c in cols if c != bgc]
    c1, c2, c3, c4 = random.sample(rem, 4)

    # the only discrete structural variant the generator has is the final
    # rotation, which decides which side of the frame the rays grow from
    VARIANTS = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "c1": c1, "c2": c2, "c3": c3, "c4": c4,
            "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, c1, c2, c3, c4, rot=None) -> dict:
    import random

    def ui(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        return random.randint(a + int((b - a) * lb), a + int((b - a) * ub))

    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    hmax = max(7, min(30, int(max_h)))
    wmax = max(7, min(30, int(max_w)))

    def line_cands(g, hh, ww, bg):
        rows, cols = [], []
        for r in range(hh):
            d = {}
            for v in g[r]:
                d[v] = d.get(v, 0) + 1
            for v, n in d.items():
                if v != bg and n >= ww - 2:
                    rows.append(r)
                    break
        for c in range(ww):
            d = {}
            for r in range(hh):
                v = g[r][c]
                d[v] = d.get(v, 0) + 1
            for v, n in d.items():
                if v != bg and n >= hh - 2:
                    cols.append(c)
                    break
        return rows, cols

    def rot_cw(m):
        return [list(r) for r in zip(*m[::-1])]

    for _outer in range(400):
        h = ui(diff_lb, diff_ub, (7, hmax))
        w = ui(diff_lb, diff_ub, (7, wmax))
        loci1 = random.randint(1, h - 4)
        locj1 = random.randint(1, w - 4)
        loci1 -= ui(diff_lb, diff_ub, (0, loci1 - 1))
        locj1 -= ui(diff_lb, diff_ub, (0, locj1 - 1))
        loci2 = ui(diff_lb, diff_ub, (loci1 + 2, h - 2))
        locj2 = ui(diff_lb, diff_ub, (locj1 + 2, w - 2))

        g = [[bgc] * w for _ in range(h)]
        fronts = [('r', loci1, c1), ('r', loci2, c2), ('c', locj1, c3), ('c', locj2, c4)]
        random.shuffle(fronts)
        for kind, idx, col in fronts:
            if kind == 'r':
                for j in range(w):
                    g[idx][j] = col
            else:
                for i in range(h):
                    g[i][idx] = col

        cands = [(i, j) for i in range(h) for j in range(w) if g[i][j] == bgc]
        nn = len(cands)
        if nn < 4:
            continue

        placed = []
        ok = False
        for _inner in range(14):
            for (i, j) in placed:
                g[i][j] = bgc
            nnoise = ui(diff_lb, diff_ub, (1, max(1, nn // 3)))
            noise = random.sample(cands, min(nnoise, nn))
            for (i, j) in noise:
                g[i][j] = c1
            placed = noise
            rws, cls = line_cands(g, h, w, bgc)
            if rws != [loci1, loci2] or cls != [locj1, locj2]:
                continue
            inner_noise = any(g[i][j] == c1
                              for i in range(loci1 + 1, loci2)
                              for j in range(locj1 + 1, locj2))
            if not inner_noise:
                continue
            ok = True
            break
        if not ok:
            continue

        H = loci2 - loci1 + 1
        W = locj2 - locj1 + 1
        base = [row[locj1:locj2 + 1] for row in g[loci1:loci2 + 1]]
        out = [r[:] for r in base]
        for j in range(W):
            m = -1
            for i in range(H):
                if base[i][j] == c1:
                    m = i
            for i in range(m + 1):
                out[i][j] = c1

        gi, go = g, out
        for _ in range(rot % 4):
            gi = rot_cw(gi)
            go = rot_cw(go)
        return {"input": tuple(tuple(int(v) for v in r) for r in gi),
                "output": tuple(tuple(int(v) for v in r) for r in go)}

    # extremely defensive fallback (should not be reached)
    h = w = 7
    g = [[bgc] * w for _ in range(h)]
    for j in range(w):
        g[1][j] = c1
        g[5][j] = c2
    for i in range(h):
        g[i][1] = c3
        g[i][5] = c4
    g[3][3] = c1
    base = [row[1:6] for row in g[1:6]]
    out = [r[:] for r in base]
    for i in range(3):
        out[i][2] = c1
    return {"input": tuple(tuple(int(v) for v in r) for r in g),
            "output": tuple(tuple(int(v) for v in r) for r in out)}


def derive_operations(I, O, examples=None):
    import numpy as np
    try:
        from maker.sel_helpers import sel_of
    except Exception:
        def sel_of(cells):
            return {"cells": [(int(a), int(b)) for a, b in cells]}

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def tally(vals):
        d = {}
        for v in vals:
            d[v] = d.get(v, 0) + 1
        return d

    def line_col(vals, bg, need):
        for v, n in tally(vals).items():
            if v != bg and n >= need:
                return int(v)
        return None

    def find_frame(bg):
        # the four painted frontiers: full rows / columns of one non-background
        # colour, broken at most where the perpendicular pair crosses them
        cand_r, cand_c = [], []
        for r in range(h):
            v = line_col(I[r].tolist(), bg, w - 2)
            if v is not None:
                cand_r.append((r, v))
        for c in range(w):
            v = line_col(I[:, c].tolist(), bg, h - 2)
            if v is not None:
                cand_c.append((c, v))
        best = None
        for a in range(len(cand_r)):
            for b in range(a + 1, len(cand_r)):
                r1, vt = cand_r[a]
                r2, vb = cand_r[b]
                if r2 - r1 < 2:
                    continue
                for x in range(len(cand_c)):
                    for y in range(x + 1, len(cand_c)):
                        cc1, vl = cand_c[x]
                        cc2, vr = cand_c[y]
                        if cc2 - cc1 < 2:
                            continue
                        four = [vt, vb, vl, vr]
                        if len(set(four)) != 4:
                            continue
                        ok = True
                        for rr, vv in ((r1, vt), (r2, vb)):
                            for j in range(w):
                                if int(I[rr, j]) != vv and j not in (cc1, cc2):
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            for ccx, vv in ((cc1, vl), (cc2, vr)):
                                for i in range(h):
                                    if int(I[i, ccx]) != vv and i not in (r1, r2):
                                        ok = False
                                        break
                                if not ok:
                                    break
                        if not ok:
                            continue
                        pal = set(I[r1 + 1:r2, cc1 + 1:cc2].flatten().tolist()) - {bg}
                        if len(pal) > 1:
                            continue
                        if len(pal) == 1 and int(next(iter(pal))) not in four:
                            continue
                        score = ((r2 - r1) * (cc2 - cc1), -r1, -cc1)
                        if best is None or score > best[0]:
                            best = (score, (r1, r2, cc1, cc2, vt, vb, vl, vr))
        return best

    bgc, frame = None, None
    for cand_bg, _n in sorted(tally(I.flatten().tolist()).items(),
                              key=lambda kv: (-kv[1], kv[0])):
        f = find_frame(int(cand_bg))
        if f is not None:
            bgc = int(cand_bg)
            frame = f
            break

    if frame is None:                       # defensive: nothing recognisable
        return [34], [[0, 0, h - 1, w - 1]]

    r1, r2, cc1, cc2, vt, vb, vl, vr = frame[1]
    H = r2 - r1 + 1
    W = cc2 - cc1 + 1
    sub = I[r1:r2 + 1, cc1:cc2 + 1]

    # --- the rule -------------------------------------------------------
    # keep the rectangle framed by the four lines; inside it, every speckle
    # grows into a ray aimed at the frame side that carries the speckles'
    # own colour, the ray running from that side out to the farthest speckle.
    pal = sorted(set(sub[1:H - 1, 1:W - 1].flatten().tolist()) - {bgc})
    nc = int(pal[0]) if len(pal) == 1 else None
    side = None
    if nc is not None:
        if vt == nc:
            side = 'up'
        elif vb == nc:
            side = 'down'
        elif vl == nc:
            side = 'left'
        elif vr == nc:
            side = 'right'

    rays = []          # (index of the ray's line, cells of the ray)
    n_noise = 0
    if side is not None:
        if side in ('up', 'down'):
            for j in range(1, W - 1):
                hits = [i for i in range(1, H - 1) if int(sub[i, j]) == nc]
                n_noise += len(hits)
                if not hits:
                    continue
                if side == 'up':
                    seg = [(i, j) for i in range(1, max(hits) + 1)]
                else:
                    seg = [(i, j) for i in range(min(hits), H - 1)]
                if all(int(sub[a, b]) == nc for a, b in seg):
                    continue      # this ray is already drawn by the speckles
                rays.append((j, seg))
        else:
            for i in range(1, H - 1):
                hits = [j for j in range(1, W - 1) if int(sub[i, j]) == nc]
                n_noise += len(hits)
                if not hits:
                    continue
                if side == 'left':
                    seg = [(i, j) for j in range(1, max(hits) + 1)]
                else:
                    seg = [(i, j) for j in range(min(hits), W - 1)]
                if all(int(sub[a, b]) == nc for a, b in seg):
                    continue
                rays.append((i, seg))

    # --- route choices, all read off the input -------------------------
    ok_nc = nc if nc is not None else 0
    order_key = (ok_nc + int(vt) + int(vl) + len(rays)) % 4
    if order_key == 0:                       # sweep along the frame
        rays.sort(key=lambda t: t[0])
    elif order_key == 1:                     # sweep back the other way
        rays.sort(key=lambda t: -t[0])
    elif order_key == 2:                     # longest reach first
        rays.sort(key=lambda t: (-len(t[1]), t[0]))
    else:                                    # shortest reach first
        rays.sort(key=lambda t: (len(t[1]), t[0]))

    group = (len(rays) >= 3 and (H + W + n_noise) % 3 == 0)
    if group:                                # draw all rays of equal reach together
        order, buckets = {}, []
        for _k, seg in rays:
            L = len(seg)
            if L not in order:
                order[L] = len(buckets)
                buckets.append([])
            buckets[order[L]].extend(seg)
        strokes = buckets
    else:
        strokes = [seg for _k, seg in rays]

    crop_first = ((r1 + cc1 + bgc) % 2 == 0)

    ops, sels = [], []
    # the crop selection is exactly the full framed rectangle (background
    # included), so the bbox form is exact here
    crop_sel = [r1, cc1, H - 1, W - 1]

    if crop_first:
        ops.append(33)
        sels.append(crop_sel)
        for seg in strokes:
            ops.append(int(ok_nc))           # Color(nc); nc may legitimately be 0
            sels.append(sel_of(seg))
    else:
        for seg in strokes:                  # draw the rays where they stand,
            ops.append(int(ok_nc))           # then cut the rectangle out
            sels.append(sel_of([(a + r1, b + cc1) for a, b in seg]))
        ops.append(33)
        sels.append(crop_sel)

    ops.append(34)
    sels.append([0, 0, H - 1, W - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5daaa586"
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
                                f"for task 5daaa586"
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
                    f"Failed to build a complete episode for task 5daaa586 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5daaa586-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
