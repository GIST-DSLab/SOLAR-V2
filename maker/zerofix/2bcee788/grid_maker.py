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

from maker.sel_helpers import sel_of

# ---------------------------------------------------------------- helpers ----

TRANSFORMS = ('identity', 'dmirror', 'cmirror', 'vmirror',
              'hmirror', 'rot90', 'rot180', 'rot270')

SIDES = ('right', 'left', 'bottom', 'top')


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    lo = max(lo, a)
    hi = min(hi, b)
    return random.randint(lo, hi)


def _apply_fn(name, g):
    if name == 'dmirror':
        return np.transpose(g)
    if name == 'cmirror':
        return np.rot90(np.transpose(g), 2)
    if name == 'vmirror':
        return np.fliplr(g)
    if name == 'hmirror':
        return np.flipud(g)
    if name == 'rot90':
        return np.rot90(g, 3)
    if name == 'rot180':
        return np.rot90(g, 2)
    if name == 'rot270':
        return np.rot90(g, 1)
    return g


def _side_from_cells(obj_rc, sep_rc):
    """Which side of the blob's bbox does the 1-wide marker line sit on?"""
    if set(r for r, _ in obj_rc) & set(r for r, _ in sep_rc):   # shares rows -> vertical line
        return 'right' if min(c for _, c in sep_rc) > min(c for _, c in obj_rc) else 'left'
    return 'bottom' if min(r for r, _ in sep_rc) > min(r for r, _ in obj_rc) else 'top'


def _probe(fns):
    """Where a composition of the mirror-ops puts the line, and whether it swaps dims."""
    p = np.zeros((5, 7), dtype=int)
    p[1:4, 1:3] = 1          # "blob"
    p[1:4, 3] = 2            # "marker line", built on its right
    q = p
    for fn in fns:
        q = _apply_fn(fn, q)
    obj = [(int(r), int(c)) for r, c in zip(*np.where(q == 1))]
    sep = [(int(r), int(c)) for r, c in zip(*np.where(q == 2))]
    return _side_from_cells(obj, sep), (q.shape != p.shape)


def _grow_shape(h, w, diff_lb, diff_ub):
    """Grow the blob exactly like the RE-ARC generator: a connected set that
    contains one cell of the last column and is at least 2 columns wide."""
    inds = [(i, j) for i in range(h) for j in range(w)]
    sp = (random.randint(0, h - 1), w - 1)
    shp = {sp}
    numcellsd = _unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = random.choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(inds) - {sp}

    def cands():
        nb = set()
        for (i, j) in shp:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        nb.add((i + di, j + dj))
        return sorted((reminds - shp) & nb)

    for _ in range(numc):
        cs = cands()
        if not cs:
            break
        shp.add(random.choice(cs))
    guard = 0
    while len({j for _, j in shp}) == 1 and guard <= h * w:
        cs = cands()
        if not cs:
            return None
        shp.add(random.choice(cs))
        guard += 1
    return shp if len({j for _, j in shp}) > 1 else None


# ------------------------------------------- 1. colors + per-instance plan ----

def sample_colors(num_examples=None) -> dict:
    # 3 is reserved: it is the color the background turns into.
    # 0 stays an ordinary picture color and may land on any role.
    cols = [c for c in range(10) if c != 3]
    bgc, sepc, objc = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    variants = [{"side": s} for s in SIDES]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "sepc": sepc, "objc": objc, "instance_plan": plan}


# ------------------------------------------------------------ 2. generator ----

def generate(diff_lb, diff_ub, max_h, max_w, bgc, sepc, objc, side=None) -> dict:
    if side is None or side not in SIDES:
        side = random.choice(SIDES)

    for _ in range(600):
        fns = random.sample(TRANSFORMS, random.choice((1, 2)))
        got_side, swaps = _probe(fns)
        if got_side != side:
            continue

        H_lim = max_w if swaps else max_h        # limits BEFORE the transform
        W_lim = max_h if swaps else max_w
        hmax = min(20, H_lim - 1)
        wmax = min(10, (W_lim - 1) // 2)
        if hmax < 2 or wmax < 2:
            continue

        h = _unifint(diff_lb, diff_ub, (2, hmax))
        w = _unifint(diff_lb, diff_ub, (2, wmax))
        shp = _grow_shape(h, w, diff_lb, diff_ub)
        if shp is None:
            continue

        c2 = np.full((h, w), bgc, dtype=int)         # the blob
        c3 = np.full((h, w), bgc, dtype=int)         # only its last-column cells
        for (i, j) in shp:
            c2[i, j] = objc
            if j == w - 1:
                c3[i, j] = sepc

        gimini = np.hstack([c2, np.fliplr(c3)])      # blob + marker line
        gomini = np.hstack([c2, np.fliplr(c2)])      # blob + its reflection

        fullh = _unifint(diff_lb, diff_ub, (h + 1, H_lim))
        fullw = _unifint(diff_lb, diff_ub, (2 * w + 1, W_lim))
        loci = random.randint(0, fullh - h)
        locj = random.randint(0, fullw - 2 * w)

        gi = np.full((fullh, fullw), bgc, dtype=int)
        go = np.full((fullh, fullw), bgc, dtype=int)
        gi[loci:loci + h, locj:locj + 2 * w] = gimini
        go[loci:loci + h, locj:locj + 2 * w] = gomini

        for fn in fns:
            gi = _apply_fn(fn, gi)
            go = _apply_fn(fn, go)
        go = np.where(go == bgc, 3, go)

        return {"input": gi.tolist(), "output": go.tolist()}

    raise ValueError("could not build an instance within the given bounds")


# ---------------------------------------------------- 3. ARCLE trajectory ----

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- the three roles ----------------------------------------------------
    # background = the color whose bounding box spans the picture (it always
    # owns a full row and a full column); of the other two the blob is the
    # bigger one, the 1-wide mirror marker the smaller.
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

    # the marker line tells us which side the reflection is built on
    side = _side_from_cells(obj_rc, sep_rc)
    if side in ('right', 'left'):
        flip_op = 26                                   # FlipH: left <-> right
        dj = ow if side == 'right' else -ow
        dest_r, dest_c = r0, c0 + dj
        mirror_rc = [(r, c0 + c1 - c + dj) for r, c in obj_rc]
    else:
        flip_op = 27                                   # FlipV: up <-> down
        di = oh if side == 'bottom' else -oh
        dest_r, dest_c = r0 + di, c0
        mirror_rc = [(r0 + r1 - r + di, c) for r, c in obj_rc]

    # --- 1. the picture's background becomes 3 (base layer) -----------------
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(I == bgc))]
    if bgc != 3 and bg_cells:
        ops.append(3)
        sels.append(sel_of(bg_cells))

    if objc != 0:
        # --- 2. duplicate the blob's whole bbox (blob + its 3-background) ---
        # bbox selection intended here: the whole rectangle, background included.
        ops.append(29)                                 # CopyO (grid already recolored)
        sels.append([r0, c0, oh - 1, ow - 1])
        # --- 3. drop that rectangle on the far side of the marker line -----
        ops.append(30)                                 # Paste at destination origin
        sels.append([dest_r, dest_c, 0, 0])
        # --- 4. reflect it in place: that is the rule ----------------------
        src = np.where(I[r0:r1 + 1, c0:c1 + 1] == bgc, 3, I[r0:r1 + 1, c0:c1 + 1])
        mirrored = np.fliplr(src) if flip_op == 26 else np.flipud(src)
        if not np.array_equal(src, mirrored):          # skip only a literal no-op
            ops.append(flip_op)
            sels.append([dest_r, dest_c, oh - 1, ow - 1])  # whole rectangle, on purpose
    else:
        # The blob is drawn in color 0, which Copy/Paste read as "nothing there",
        # so a copy would arrive empty: paint the reflected blob's own cells
        # instead. They cover the whole marker line, so the line goes with it.
        ops.append(0)
        sels.append(sel_of(mirror_rc))

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
