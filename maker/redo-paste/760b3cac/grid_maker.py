"""
ARC Task: 760b3cac (RE-ARC) — LLM-generated grid_maker
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
    # object colour must be non-zero: the reflected copy is produced with CopyI/Paste,
    # and ARCLE's clipboard never carries colour 0.
    objc = random.choice([c for c in cols if c != bgc and c != 0])
    indc = random.choice([c for c in cols if c not in (bgc, objc)])

    # discrete structural variant: the direction the arrow-marker points
    dirs = ["U", "D", "L", "R"]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(dirs):
        examples = [{"direction": d} for d in dirs]
        examples += [{"direction": random.choice(dirs)} for _ in range(n_ex - len(dirs))]
        random.shuffle(examples)
    else:
        examples = [{"direction": d} for d in random.sample(dirs, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "objc": objc, "indc": indc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, objc, indc, direction=None) -> dict:
    if direction is None:
        direction = choice(("U", "D", "L", "R"))

    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)

    def is_arrow(cells):
        cells = frozenset(cells)
        if len(cells) != 5:
            return False
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        if max(rs) - r0 != 2 or max(cs) - c0 != 2:
            return False
        corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
        peri = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)} - {(r0 + 1, c0 + 1)}
        return len(cells & peri) == 4 and len(cells & corners) == 1

    # after rotation the canvas is transposed for the U/D variants
    rotated = direction in ("U", "D")
    hlim = min(30, max_w if rotated else max_h)
    wlim = min(29, max_h if rotated else max_w)
    kmax = min(14, (wlim - 1) // 2)
    if hlim < 5 or kmax < 3:
        raise ValueError("grid size limits too small for this task")

    h = unifint(diff_lb, diff_ub, (5, hlim))
    w = 2 * unifint(diff_lb, diff_ub, (3, kmax)) + 1

    if direction == "R":
        sgn, rotf = choice(((1, identity), (-1, rot180)))
    elif direction == "L":
        sgn, rotf = choice(((-1, identity), (1, rot180)))
    elif direction == "D":
        sgn, rotf = choice(((1, rot90), (-1, rot270)))
    else:
        sgn, rotf = choice(((-1, rot90), (1, rot270)))
    obj = objR if sgn == 1 else objL

    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    # keep the reflected copy fully inside the canvas: objw <= floor(w/3), odd
    wcap = w // 3
    if wcap % 2 == 0:
        wcap -= 1
    kw = min(w // 6, (wcap - 1) // 2)
    objw = 2 * unifint(diff_lb, diff_ub, (1, kw)) + 1 if kw >= 1 else 1

    cnv = canvas(-1, (objh, objw))
    inds = asindices(cnv)
    objx = None
    for _try in range(64):
        sp = choice(totuple(inds))
        cur = {sp}
        numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
        numc = choice((numcd, objh * objw - numcd))
        numc = min(max(1, numc), objh * objw)
        for _k in range(numc - 1):
            cur.add(choice(totuple((inds - cur) & mapply(neighbors, cur))))
        while width(cur) != objw:
            cur.add(choice(totuple((inds - cur) & mapply(neighbors, cur))))
        cand = normalize(frozenset(cur))
        # the payload must not look like the arrow marker itself (would be ambiguous)
        if not is_arrow(cand):
            objx = cand
            break
    if objx is None:
        raise ValueError("could not build a non-ambiguous object")

    objh, objw = shape(objx)

    gi = canvas(bgc, (h, w))
    gi = fill(gi, indc, shift(obj, (h - 3, w // 2 - 1)))
    loci = randint(0, h - 3 - objh)
    locj = w // 2 - objw // 2
    plcd = shift(objx, (loci, locj))
    gi = fill(gi, objc, plcd)
    plcd2 = shift(vmirror(plcd), (0, objw * sgn))
    go = fill(gi, objc, plcd2)

    return {'input': rotf(gi), 'output': rotf(go)}


def derive_operations(I, O):
    import numpy as np
    from collections import Counter

    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def bbox(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1

    def is_arrow(cells):
        if len(cells) != 5:
            return False
        r0, c0, h, w = bbox(cells)
        if h != 3 or w != 3:
            return False
        s = set(cells)
        corners = {(r0, c0), (r0, c0 + 2), (r0 + 2, c0), (r0 + 2, c0 + 2)}
        peri = {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)} - {(r0 + 1, c0 + 1)}
        return len(s & peri) == 4 and len(s & corners) == 1

    def direction_of(cells):
        # the marker's "wide" side (2 cells on one bbox edge) is where it points;
        # pointing up/down -> mirror up-down (FlipV, 27), left/right -> FlipH (26)
        r0, c0, h, w = bbox(cells)
        if sum(1 for r, c in cells if r == r0 + h - 1) == 2:
            return 1, 0, 27
        if sum(1 for r, c in cells if r == r0) == 2:
            return -1, 0, 27
        if sum(1 for r, c in cells if c == c0 + w - 1) == 2:
            return 0, 1, 26
        return 0, -1, 26

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    parts = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                parts.setdefault(v, []).append((r, c))
    cols = list(parts.keys())

    arrows = [v for v in cols if is_arrow(parts[v])]
    if len(arrows) == 1:
        ind_col = arrows[0]
    else:
        cands = arrows if arrows else cols
        ind_col = None
        for a in cands:
            rest = [v for v in cols if v != a]
            if len(rest) != 1:
                continue
            oc = rest[0]
            dr, dc, _f = direction_of(parts[a])
            r0, c0, oh, ow = bbox(parts[oc])
            good = True
            for (r, c) in parts[oc]:
                rr = (2 * r0 + oh - 1 - r) if dr else r
                cc = (2 * c0 + ow - 1 - c) if dc else c
                rr += dr * oh
                cc += dc * ow
                if 0 <= rr < ho and 0 <= cc < wo and O[rr, cc] != oc:
                    good = False
                    break
            if good:
                ind_col = a
                break
        if ind_col is None:
            ind_col = cands[0]

    obj_col = [v for v in cols if v != ind_col][0]
    dr, dc, fop = direction_of(parts[ind_col])
    r0, c0, oh, ow = bbox(parts[obj_col])

    # destination = the object's own rectangle reflected across the edge it points at,
    # i.e. translated by exactly one object height/width in that direction
    d_r0, d_c0 = r0 + dr * oh, c0 + dc * ow
    s_r0, s_c0, s_h, s_w = r0, c0, oh, ow
    if dr == 1:
        m = min(oh, ho - d_r0)
        s_r0, s_h = r0 + oh - m, m
    elif dr == -1:
        m = min(oh, d_r0 + oh)
        s_h, d_r0 = m, max(d_r0, 0)
    if dc == 1:
        m = min(ow, wo - d_c0)
        s_c0, s_w = c0 + ow - m, m
    elif dc == -1:
        m = min(ow, d_c0 + ow)
        s_w, d_c0 = m, max(d_c0, 0)

    ops, sels = [], []
    if s_h > 0 and s_w > 0:
        # CopyI: the object's whole rectangle from the INPUT (background included).
        # bbox selection is exact here: the intended cells ARE that full rectangle.
        ops.append(28)
        sels.append([s_r0, s_c0, s_h - 1, s_w - 1])
        # Paste it at the reflected position (only the top-left corner is used as origin)
        ops.append(30)
        sels.append([d_r0, d_c0, 0, 0])
        # Mirror the pasted rectangle in place -> the reflection the marker asks for.
        # Full-rectangle selection again; skipped when the block is already symmetric
        # (the op would be an invisible no-op).
        blk = I[s_r0:s_r0 + s_h, s_c0:s_c0 + s_w]
        mir = np.fliplr(blk) if fop == 26 else np.flipud(blk)
        if not np.array_equal(blk, mir):
            ops.append(fop)
            sels.append([d_r0, d_c0, s_h - 1, s_w - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 760b3cac"
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
                                f"for task 760b3cac"
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
                    f"Failed to build a complete episode for task 760b3cac "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"760b3cac-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
