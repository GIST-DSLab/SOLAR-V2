"""
ARC Task: e179c5f4 (RE-ARC) — LLM-generated grid_maker
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

# The code-golf entry reads this task differently from the maker. It does not follow
# one ray bounce by bounce and then mirror whole rays onto the other markers. It
# treats the output as a pattern that repeats along the long axis: the bounce comes
# back every P = 2*(short-1) steps. Each marker's ray is periodic with that period,
# so their union is periodic too. So the derivation below draws ONE period band
# (every marker's straight strokes inside it), then copies that band down the long
# axis. Where the two readings agree, nothing changes: the background becomes 8 with
# one FloodFill8 seed.


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    if bgc is None or linc is None:
        bgc, linc = random.sample([c for c in range(10) if c != 8], 2)

    def unif(lo, hi):
        if hi < lo:
            hi = lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, lo + round(d * (hi - lo))))

    cap = min(30, max_h, max_w)          # transforms may transpose, so bound both dims
    w = unif(2, min(10, cap - 1))
    h = unif(w + 1, cap)
    gi = [[bgc] * w for _ in range(h)]
    gi[h - 1][0] = linc
    go = [row[:] for row in gi]
    r, c, d = h - 1, 0, 1
    while True:
        r, c = r - 1, c + d
        if c == w - 1 or c == 0:
            d *= -1
        if r < 0:
            break
        go[r][c] = linc
    gi = np.array(gi)
    go = np.array(go)
    mfs = [lambda g: g, lambda g: g.T, lambda g: np.rot90(g, 2).T,
           np.fliplr, np.flipud, lambda g: np.rot90(g, 3),
           lambda g: np.rot90(g, 2), lambda g: np.rot90(g, 1)]
    for fn in random.sample(mfs, random.choice((1, 2))):
        gi = fn(gi)
        go = fn(go)
    gix, gox = gi.copy(), go.copy()
    numlins = unif(1, 4)
    extra = []
    if numlins > 1:
        extra.append(np.flipud)
    if numlins > 2:
        extra.append(np.fliplr)
    if numlins > 3:
        extra.append(lambda g: np.flipud(np.fliplr(g)))
    for fn in extra:
        gi = np.where(fn(gix) == linc, linc, gi)
        go = np.where(fn(gox) == linc, linc, go)
    go = np.where(go == bgc, 8, go)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    H, W = I.shape
    portrait = H >= W
    L, S = (H, W) if portrait else (W, H)
    P = 2 * (S - 1)                          # the zigzag repeats every P steps along the long axis

    def rc(a, b):                            # (long, short) axis -> grid (r, c)
        return (a, b) if portrait else (b, a)

    bseed = (1, 0) if H >= 3 else (0, 1)     # never a corner, so always background
    bgc = int(I[bseed])
    linc = [int(v) for v in np.unique(I) if int(v) != bgc][0]

    corners = sorted({(r, c) for r in (0, H - 1) for c in (0, W - 1) if I[r, c] == linc})

    def tri(k):
        m = k % P
        return m if m <= S - 1 else P - m

    def ray(corner):                         # ordered cells, one per step along the long axis
        a0, b0 = corner if portrait else (corner[1], corner[0])
        sa = 1 if a0 == 0 else -1
        out = []
        for k in range(L):
            t = tri(k)
            b = t if b0 == 0 else S - 1 - t
            out.append((k, rc(a0 + sa * k, b)))
        return out

    ops, sels = [], []
    cur = I.copy()

    # 1) background -> 8 (one connected region, one seed)
    ops.append(18)
    sels.append(sel_of([bseed]))
    cur[cur == bgc] = 8

    # 2) draw ONE period of the bouncing pattern: the first P steps of the long axis.
    #    Each marker's ray contributes its straight diagonal strokes inside that band.
    T = min(P, L)

    def in_band(p):
        a = p[0] if portrait else p[1]
        return a < T

    near = [cn for cn in corners if (cn[0] if portrait else cn[1]) == 0]
    far = [cn for cn in corners if cn not in near]
    for cn in near + far:
        segs = {}
        for k, p in ray(cn):
            if in_band(p):
                segs.setdefault(k // (S - 1), []).append(p)
        for j in sorted(segs):               # one straight stroke per bounce
            cells = [p for p in segs[j] if cur[p] != linc]
            if cells:
                ops.append(linc)
                sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = linc

    # 3) the pattern is periodic with period P: repeat the drawn band along the long axis
    def band_rect(a0, n):                    # full rectangle of n long-axis steps from a0
        if portrait:
            return [a0, 0, n - 1, W - 1]
        return [0, a0, H - 1, n - 1]

    copied = None
    off = P
    while off < L:
        n = min(P, L - off)                  # last band may be partial
        if linc != 0:
            tgt = cur.copy()
            for a in range(n):
                if portrait:
                    tgt[off + a, :] = cur[a, :]
                else:
                    tgt[:, off + a] = cur[:, a]
            if not np.array_equal(tgt, cur):     # band already in place -> nothing to paste
                if copied != n:
                    ops.append(29)               # CopyO the drawn band (full rectangle intended)
                    sels.append(band_rect(0, n))
                    copied = n
                ops.append(30)
                sels.append([off, 0, 0, 0] if portrait else [0, off, 0, 0])
                cur = tgt
        else:
            # colour 0 is transparent to Paste -> paint the cells the band copy lands on
            cells = []
            for a in range(n):
                for b in range(S):
                    src, dst = rc(a, b), rc(off + a, b)
                    if cur[src] == linc and cur[dst] != linc:
                        cells.append(dst)
            if cells:
                ops.append(linc)
                sels.append(sel_of(cells))
                for p in cells:
                    cur[p] = linc
        off += P

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
                        f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e179c5f4"
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
                                f"for task e179c5f4"
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
                    f"Failed to build a complete episode for task e179c5f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e179c5f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
