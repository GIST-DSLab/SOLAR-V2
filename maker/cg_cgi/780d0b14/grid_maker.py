"""
ARC Task: 780d0b14 (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    lo = a + int(round((b - a) * lb))
    hi = a + int(round((b - a) * ub))
    lo = max(a, min(b, lo))
    hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


def _bands(n, fl):
    res, start = [], None
    for i in range(n):
        if i in fl:
            if start is not None:
                res.append((start, i - 1))
                start = None
        elif start is None:
            start = i
    if start is not None:
        res.append((start, n - 1))
    return res


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    remcols = [c for c in range(10) if c != bgc]
    for _attempt in range(200):
        nh = _unifint(diff_lb, diff_ub, (2, 6))
        nw = _unifint(diff_lb, diff_ub, (2, 6))
        if 3 * nh - 1 > max_h or 3 * nw - 1 > max_w:
            continue
        ncols = _unifint(diff_lb, diff_ub, (3, 9))
        ccols = random.sample(remcols, ncols)
        while True:
            go = [[random.choice(ccols) for _ in range(nw)] for _ in range(nh)]
            rows = set(tuple(r) for r in go)
            cols = set(tuple(go[i][j] for i in range(nh)) for j in range(nw))
            # dedupe in DSL removes consecutive duplicates; require distinct neighbours
            okr = all(go[i] != go[i + 1] for i in range(nh - 1))
            okc = all([go[i][j] for i in range(nh)] != [go[i][j + 1] for i in range(nh)]
                      for j in range(nw - 1))
            if okr and okc:
                break
        h = _unifint(diff_lb, diff_ub, (3 * nh - 1, max_h))
        w = _unifint(diff_lb, diff_ub, (3 * nw - 1, max_w))
        hdist = [2] * nh
        for _ in range(h - 3 * nh + 1):
            hdist[random.randint(0, nh - 1)] += 1
        wdist = [2] * nw
        for _ in range(w - 3 * nw + 1):
            wdist[random.randint(0, nw - 1)] += 1
        gi = np.full((h, w), bgc, dtype=int)
        rpos, r = [], 0
        for k in range(nh):
            rpos.append((r, r + hdist[k]))
            r += hdist[k] + 1
        cpos, c = [], 0
        for k in range(nw):
            cpos.append((c, c + wdist[k]))
            c += wdist[k] + 1
        for i, (r0, r1) in enumerate(rpos):
            for j, (c0, c1) in enumerate(cpos):
                gi[r0:r1, c0:c1] = go[i][j]
                cells = [(rr, cc) for rr in range(r0, r1) for cc in range(c0, c1)]
                k = _unifint(diff_lb, diff_ub, (1, len(cells) // 2))
                for (rr, cc) in random.sample(cells, k):
                    gi[rr, cc] = bgc
        # reject ambiguous grids: frontier structure must be recoverable from I
        frow = set(rr for rr in range(h) if all(gi[rr, :] == bgc))
        fcol = set(cc for cc in range(w) if all(gi[:, cc] == bgc))
        if len(_bands(h, frow)) != nh or len(_bands(w, fcol)) != nw:
            continue
        ur = [rr for rr in range(h) if len(set(gi[rr].tolist())) == 1 and gi[rr, 0] != bgc]
        uc = [cc for cc in range(w) if len(set(gi[:, cc].tolist())) == 1 and gi[0, cc] != bgc]
        if ur or uc:
            continue
        return {"input": gi.tolist(), "output": [list(r) for r in go]}
    raise ValueError("generation failed")


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    def uniform_colors(G):
        G = np.asarray(G, dtype=int)
        rs = set(int(G[r, 0]) for r in range(G.shape[0]) if len(set(G[r].tolist())) == 1)
        cs = set(int(G[0, c]) for c in range(G.shape[1]) if len(set(G[:, c].tolist())) == 1)
        return rs & cs

    # background = the colour forming full-length lines in both directions,
    # confirmed against the demonstration inputs (same background per episode)
    cand = uniform_colors(I)
    if examples:
        for ex in examples:
            ei = ex[0] if not isinstance(ex, dict) else ex["input"]
            c2 = cand & uniform_colors(ei)
            if c2:
                cand = c2
    bgc = sorted(cand)[0]

    frow = set(r for r in range(hi) if all(I[r, :] == bgc))
    fcol = set(c for c in range(wi) if all(I[:, c] == bgc))
    rbands = _bands(hi, frow)
    cbands = _bands(wi, fcol)
    nh, nw = len(rbands), len(cbands)

    ops, sels = [], []
    # Each block (row-band x column-band) is one unit holding one colour;
    # it collapses onto cell (i, j) of the top-left nh x nw corner.
    for i, (r0, r1) in enumerate(rbands):
        for j, (c0, c1) in enumerate(cbands):
            blk = I[r0:r1 + 1, c0:c1 + 1]
            fg = int(next(iter(set(blk.flatten().tolist()) - {bgc})))
            if int(I[i, j]) != fg:
                ops.append(fg)
                sels.append(sel_of([(i, j)]))

    # keep only the collapsed block grid (full rectangle crop)
    ops.append(33)
    sels.append([0, 0, nh - 1, nw - 1])
    ops.append(34)
    sels.append([0, 0, nh - 1, nw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 780d0b14"
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
                                f"for task 780d0b14"
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
                    f"Failed to build a complete episode for task 780d0b14 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"780d0b14-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
