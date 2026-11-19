"""
ARC Task: 53b68214 (RE-ARC) — LLM-generated grid_maker
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
VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, 9)
    ccols = random.sample(remcols, ncols)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, mirror=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if ccols is None:
        remcols = remove(bgc, cols)
        ccols = sample(remcols, unifint(diff_lb, diff_ub, (1, 9)))
    ccols = list(ccols)
    if mirror is None:
        mirror = choice((True, False))
    if max_h < 10:
        raise ValueError("output needs 10 rows")
    wmax = max(8, min(30, max_w))
    while True:
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (8, wmax))
        oh = unifint(diff_lb, diff_ub, (1, h // 2))
        ow = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w // 2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (10, w))
        hoffs = randint(0, ow // 2 + 1)
        for k in range(10 // oh + 1):
            go = paint(go, shift(plcd, (k * oh, k * hoffs)))
        if len(palette(go[h:])) > 1:
            break
    gi = go[:h]
    if mirror:
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O=None, examples=None):
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    H_OUT = 10                      # the rule always continues the pattern to 10 rows
    I = np.asarray(I, dtype=int)
    h, w = I.shape

    def bg_of(G):
        # the motif covers < half of every row, so the background is the majority colour
        return int(Counter(np.asarray(G).flatten().tolist()).most_common(1)[0][0])

    def consistent(G, bgc, t, k):
        # reading: row r is the first band's row r % t, slid sideways by (r // t) * k
        hh, ww = G.shape
        for r in range(t, hh):
            j, rr = divmod(r, t)
            s = j * k
            for c in range(ww):
                sc = c - s
                exp = G[rr, sc] if 0 <= sc < ww else bgc
                if G[r, c] != exp:
                    return False
        return True

    def candidates(G, bgc):
        hh, ww = G.shape
        out = []
        for t in range(1, max(1, hh // 2) + 1):
            for k in range(-(ww - 1), ww):
                if consistent(G, bgc, t, k):
                    out.append((t, k))
        return out

    def extend(G, bgc, t, k):
        hh, ww = G.shape
        P = np.full((H_OUT, ww), bgc, dtype=int)
        for r in range(H_OUT):
            j, rr = divmod(r, t)
            for c in range(ww):
                sc = c - j * k
                if 0 <= sc < ww:
                    P[r, c] = G[rr, sc]
        P[:hh] = G
        return P

    POLICIES = [
        lambda tk: (tk[0], abs(tk[1]), -tk[1]),
        lambda tk: (abs(tk[1]), tk[0], -tk[1]),
        lambda tk: (-tk[0], abs(tk[1]), -tk[1]),
    ]

    def pick(G, bgc, pol):
        cs = candidates(G, bgc)
        if not cs:
            return (max(1, G.shape[0] // 2), 0)
        return min(cs, key=POLICIES[pol])

    # which tie-break among I-consistent periods the demonstrations follow
    pol = 0
    if examples:
        for p in range(len(POLICIES)):
            ok = True
            for ei, eo in examples:
                ei = np.asarray(ei, dtype=int)
                eo = np.asarray(eo, dtype=int)
                b = bg_of(ei)
                if not np.array_equal(extend(ei, b, *pick(ei, b, p)), eo):
                    ok = False
                    break
            if ok:
                pol = p
                break

    bgc = bg_of(I)
    t, k = pick(I, bgc, pol)

    # ---- plan the trajectory on a simulated working grid ----
    def build(with_base):
        W = np.zeros((H_OUT, w), dtype=int)
        W[:h] = I
        clip = None
        ops, sels = [], []
        # 1. grow the canvas to 10 rows (full rectangle)
        ops.append(33); sels.append([0, 0, H_OUT - 1, w - 1])
        # 2. background base under the added rows (full rectangle)
        if with_base:
            ops.append(int(bgc)); sels.append([h, 0, H_OUT - 1 - h, w - 1])
            W[h:, :] = bgc
        # 3. stamp the first band again at every multiple of (t, k)
        j = 1
        while j * t < H_OUT:
            r0 = j * t
            nrows = min(t, H_OUT - r0)
            if r0 + nrows <= h:
                j += 1
                continue
            s = j * k
            if s >= 0:
                c0, cw, dst = 0, w - s, s
            else:
                c0, cw, dst = -s, w + s, 0
            if cw <= 0:
                j += 1
                continue
            src = I[0:nrows, c0:c0 + cw]
            tgt = W[r0:r0 + nrows, dst:dst + cw]
            new = np.where(src > 0, src, tgt)
            if not np.array_equal(new, tgt):
                key = (nrows, c0, cw)
                if clip != key:
                    # the part of the first band that stays on the grid after the slide
                    ops.append(28); sels.append([0, c0, nrows - 1, cw - 1])
                    clip = key
                ops.append(30); sels.append([r0, dst, 0, 0])
                W[r0:r0 + nrows, dst:dst + cw] = new
            # colour 0 used as a motif colour is not carried by Copy/Paste
            if bgc != 0:
                zc = [(r0 + rr, dst + cc) for rr in range(nrows) for cc in range(cw)
                      if src[rr, cc] == 0 and r0 + rr >= h and W[r0 + rr, dst + cc] != 0]
                if zc:
                    ops.append(0); sels.append(sel_of(zc))
                    for (r, c) in zc:
                        W[r, c] = 0
            j += 1
        ops.append(34); sels.append([0, 0, H_OUT - 1, w - 1])
        return ops, sels, W

    if bgc != 0:
        ops_b, sels_b, Wb = build(True)
        ops_n, sels_n, Wn = build(False)
        # the base is only needed where no stamped band reaches
        if np.array_equal(Wb, Wn):
            return ops_n, sels_n
        return ops_b, sels_b
    ops, sels, _ = build(False)
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
                        f"num_examples+1 ({num_examples + 1}) for task 53b68214"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 53b68214"
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
                                f"for task 53b68214"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O, list(zip(ex_in, ex_out)))
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task 53b68214 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"53b68214-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
