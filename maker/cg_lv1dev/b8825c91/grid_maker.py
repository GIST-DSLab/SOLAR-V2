"""
ARC Task: b8825c91 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 4]
    return {"bgc": random.choice(cols)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    def _unifint(lb, ub, rng):
        a, b = rng
        if b <= a:
            return a
        lo = a + int(round((b - a) * lb)); hi = a + int(round((b - a) * ub))
        lo, hi = max(a, min(lo, b)), max(a, min(hi, b))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = [c for c in range(10) if c != 4]
    if bgc is None:
        bgc = random.choice(cols)
    hmax = max(3, min(15, min(max_h, max_w) // 2))
    h = _unifint(diff_lb, diff_ub, (3, hmax)); w = h
    remcols = [c for c in cols if c != bgc]
    numcols = _unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    canv = np.full((h, w), bgc, dtype=int)
    nc = _unifint(diff_lb, diff_ub, (1, h * w))
    bx = {(i, j) for i in range(h) for j in range(w)}
    start = random.choice(sorted(bx))
    obj = {start: random.choice(remcols)}
    for _ in range(nc - 1):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx -= {ch}
    for (i, j), v in obj.items():
        canv[i, j] = v
    t = canv.T.copy()
    for i in range(h):
        for j in range(w):
            if j >= i:
                canv[i, j] = t[i, j]
    gi = np.hstack([canv, canv[:, ::-1]])
    gi = np.vstack([gi, gi[::-1, :]])
    go = gi.copy()
    for alph in (2, 1):
        loci = alph * h - _unifint(diff_lb, diff_ub, (1, alph * h))
        locj = w - _unifint(diff_lb, diff_ub, (1, w))
        loci2 = _unifint(diff_lb, diff_ub, (loci, alph * h - 1))
        locj2 = _unifint(diff_lb, diff_ub, (locj, w - 1))
        gi[loci:loci2 + 1, locj:locj2 + 1] = 4
        gi, go = np.rot90(gi, 2), np.rot90(go, 2)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T, np.fliplr, np.flipud,
           lambda g: np.rot90(g, 3), lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi, go = fn(gi), fn(go)
    return {"input": np.ascontiguousarray(gi).tolist(),
            "output": np.ascontiguousarray(go).tolist()}


def derive_operations(I, O=None, examples=None):
    # Reading (golf entry): the picture equals its own 180-degree turn, so each hidden
    # cell is read from its point-reflected partner.  The whole grid is the unit: turn
    # it 180 degrees (FlipH + FlipV on the full rectangle).  Every 4-patch is now
    # covered by its partner's content; the only 4s left are the turned images of the
    # patches, which sit where I was visible -- restore those rectangles from I in
    # place (CopyI + Paste).  If a patch's point partner was itself hidden, fall back
    # to another mirror partner of I, flipping the pasted copy as that partner needs.
    # Nothing is read from O.
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    ops, sels = [], []
    full = [0, 0, H - 1, W - 1]          # whole grid, full rectangle
    G = I.copy()
    t = np.fliplr(G)
    if not np.array_equal(t, G):
        ops.append(26); sels.append(full); G = t
    t = np.flipud(G)
    if not np.array_equal(t, G):
        ops.append(27); sels.append(full); G = t

    # remaining 4-cells (turned images of the patches) as whole rectangles
    m = (G == 4)
    rects = []
    while m.any():
        rs, cs = np.nonzero(m)
        r0, c0 = int(rs[0]), int(cs[0])
        c1 = c0
        while c1 + 1 < W and m[r0, c1 + 1]:
            c1 += 1
        r1 = r0
        while r1 + 1 < H and m[r1 + 1, c0:c1 + 1].all():
            r1 += 1
        rects.append((r0, c0, r1, c1))
        m[r0:r1 + 1, c0:c1 + 1] = False

    # source rectangle in I (top-left) and flips needed after pasting
    def partner(kind, r0, c0, r1, c1):
        if kind == 'own':
            return r0, c0, False, False
        if kind == 'lr':
            return r0, W - 1 - c1, True, False
        if kind == 'ud':
            return H - 1 - r1, c0, False, True
        return H - 1 - r1, W - 1 - c1, True, True

    leftovers = []
    work = list(rects)
    while work:
        r0, c0, r1, c1 = work.pop(0)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        pick = None
        for kind in ('own', 'lr', 'ud', 'rot'):
            sr, sc, flr, fud = partner(kind, r0, c0, r1, c1)
            if not (I[sr:sr + bh, sc:sc + bw] == 4).any():
                pick = (sr, sc, flr, fud)
                break
        if pick is None:
            if bh == 1 and bw == 1:
                leftovers.append((r0, c0))
                continue
            if bh >= bw:
                k = bh // 2
                work[:0] = [(r0, c0, r0 + k - 1, c1), (r0 + k, c0, r1, c1)]
            else:
                k = bw // 2
                work[:0] = [(r0, c0, r1, c0 + k - 1), (r0, c0 + k, r1, c1)]
            continue
        sr, sc, flr, fud = pick
        src = I[sr:sr + bh, sc:sc + bw]
        res = src
        if flr:
            res = np.fliplr(res)
        if fud:
            res = np.flipud(res)
        patch = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        if (src == 0).all():
            ops.append(0); sels.append(sel_of(patch))
            G[r0:r1 + 1, c0:c1 + 1] = 0
            continue
        ops.append(28); sels.append([sr, sc, bh - 1, bw - 1])      # partner rectangle in I
        if (src == 0).any():
            ops.append(0); sels.append(sel_of(patch))              # Paste skips 0s: clear first
        ops.append(30); sels.append([r0, c0, 0, 0])
        cur = src
        if flr and not np.array_equal(np.fliplr(cur), cur):
            ops.append(26); sels.append([r0, c0, bh - 1, bw - 1])  # the pasted rectangle
        if flr:
            cur = np.fliplr(cur)
        if fud and not np.array_equal(np.flipud(cur), cur):
            ops.append(27); sels.append([r0, c0, bh - 1, bw - 1])  # the pasted rectangle
        G[r0:r1 + 1, c0:c1 + 1] = res

    # cells hidden in all four axial partners: read the diagonal partner (the quarter
    # picture is symmetric about its diagonal), grouped by colour
    if leftovers:
        bycol = {}
        for (r, c) in leftovers:
            for (pr, pc) in ((c, r), (W - 1 - c, H - 1 - r), (c, H - 1 - r), (W - 1 - c, r)):
                if 0 <= pr < H and 0 <= pc < W and I[pr, pc] != 4:
                    bycol.setdefault(int(I[pr, pc]), []).append((r, c))
                    break
        for col in sorted(bycol):
            ops.append(col); sels.append(sel_of(bycol[col]))
    ops.append(34); sels.append(full)
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
                        f"num_examples+1 ({num_examples + 1}) for task b8825c91"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b8825c91"
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
                                f"for task b8825c91"
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
                    f"Failed to build a complete episode for task b8825c91 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b8825c91-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
