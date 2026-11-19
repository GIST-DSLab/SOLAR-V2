"""
ARC Task: 1f642eb9 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    return {"bgc": bgc, "sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqc) -> dict:
    cols = list(range(10))
    hmax = max(6, min(int(max_h), 30))
    wmax = max(6, min(int(max_w), 30))

    h = _unifint(diff_lb, diff_ub, (6, hmax))
    w = _unifint(diff_lb, diff_ub, (6, wmax))
    ih = _unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = _unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = random.randint(2, h - ih - 2)
    locj = random.randint(2, w - iw - 2)

    remcols = [c for c in cols if c != bgc and c != sqc]

    ncand = 2 * ih + 2 * iw - 4
    numcells = _unifint(diff_lb, diff_ub, (1, ncand))

    outs = []
    ins = []
    c1 = random.choice((True, False))
    c2 = random.choice((True, False))
    c3 = random.choice((True, False))
    c4 = random.choice((True, False))

    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - c3):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + c2, locj + iw - c4):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))

    locs = set(random.sample(range(ncand), numcells))
    numc = _unifint(diff_lb, diff_ub, (1, min(8, len(remcols))))
    ccols = random.sample(remcols, numc)

    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]

    gi = [[bgc for _ in range(w)] for _ in range(h)]
    for a in range(loci, loci + ih):
        for b in range(locj, locj + iw):
            gi[a][b] = sqc

    seq = [random.choice(ccols) for _ in range(len(outs))]
    for col, (a, b) in zip(seq, outs):
        gi[a][b] = col

    go = [row[:] for row in gi]
    for col, (a, b) in zip(seq, ins):
        go[a][b] = col

    return {"input": tuple(tuple(r) for r in gi),
            "output": tuple(tuple(r) for r in go)}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []
    G = I.copy()

    # background = the colour the canvas is painted with before anything is placed
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # ---- locate the solid rectangle (the block the markers gravitate onto) ----
    best = None
    for col in sorted(set(I.flatten().tolist())):
        if col == bgc:
            continue
        pts = np.argwhere(I == col)
        rr0, cc0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        rr1, cc1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        bh, bw = rr1 - rr0 + 1, cc1 - cc0 + 1
        if bh < 2 or bw < 2:
            continue                      # 1-wide marker runs are not the block
        if bh * bw != len(pts):
            continue                      # must be solid
        if best is None or bh * bw > best[0]:
            best = (bh * bw, rr0, rr1, cc0, cc1)
    _, r0, r1, c0, c1 = best

    # ---- the border markers ----
    markers = [(r, c) for r in range(hi) for c in range(wi)
               if I[r, c] != bgc and not (r0 <= r <= r1 and c0 <= c <= c1)]

    # group markers into border lines; each line replicates onto the facing edge
    edges = []
    for line in sorted({r for (r, c) in markers if r < r0 and c0 <= c <= c1}):
        ks = sorted(c for (r, c) in markers if r == line and c0 <= c <= c1)
        edges.append(('h', line, r0, ks))
    for line in sorted({r for (r, c) in markers if r > r1 and c0 <= c <= c1}):
        ks = sorted(c for (r, c) in markers if r == line and c0 <= c <= c1)
        edges.append(('h', line, r1, ks))
    for line in sorted({c for (r, c) in markers if c < c0 and r0 <= r <= r1}):
        ks = sorted(r for (r, c) in markers if c == line and r0 <= r <= r1)
        edges.append(('v', line, c0, ks))
    for line in sorted({c for (r, c) in markers if c > c1 and r0 <= r <= r1}):
        ks = sorted(r for (r, c) in markers if c == line and r0 <= r <= r1)
        edges.append(('v', line, c1, ks))

    for kind, line, dline, keys in edges:
        if kind == 'h':
            def src(k, _l=line):
                return (_l, k)

            def dst(k, _l=dline):
                return (_l, k)
        else:
            def src(k, _l=line):
                return (k, _l)

            def dst(k, _l=dline):
                return (k, _l)

        # maximal contiguous runs of non-zero marker cells; colour-0 cells stand alone
        runs = []
        for k in keys:
            col = int(I[src(k)])
            if col == 0:
                runs.append([k])
                continue
            if runs and int(I[src(runs[-1][-1])]) != 0 and k == runs[-1][-1] + 1:
                runs[-1].append(k)
            else:
                runs.append([k])

        for run in runs:
            rcol = int(I[src(run[0])])
            scells = [src(k) for k in run]
            dcells = [dst(k) for k in run]
            if rcol == 0:
                # Paste never writes colour-0 cells, so place this one explicitly
                dr, dc = dcells[0]
                if int(G[dr, dc]) != 0:
                    ops.append(0)
                    sels.append(sel_of([(dr, dc)]))
                    G[dr, dc] = 0
                continue
            # replication: copy the marker segment from the input ...
            ops.append(28)
            sels.append(sel_of(scells))          # exactly the marker run (a full 1-wide rect)
            # ... and paste it onto the block's facing edge, same row/column
            ops.append(30)
            sels.append(sel_of([dcells[0]]))     # paste origin = run's top-left destination
            for (sr, sc), (dr, dc) in zip(scells, dcells):
                G[dr, dc] = I[sr, sc]

    ops.append(34)
    sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1f642eb9"
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
                                f"for task 1f642eb9"
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
                    f"Failed to build a complete episode for task 1f642eb9 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1f642eb9-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
