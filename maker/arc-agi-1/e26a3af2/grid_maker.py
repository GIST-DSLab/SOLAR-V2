"""
ARC Task: e26a3af2 (RE-ARC) — LLM-generated grid_maker
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

# ---------------------------------------------------------------- helpers

def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


def _mostcommon(vals):
    return Counter(list(vals)).most_common(1)[0][0]


# ---------------------------------------------------------------- 1. colors

# The only discrete structural variant of this task is the orientation of the
# stripes: the generator transposes (dmirror) the pair with probability 1/2,
# turning row-bands into column-bands.  The rule itself does not depend on any
# particular colour (each band is painted with its own majority colour), so no
# colour role needs to be fixed - but BOTH orientations must be visible in the
# examples, otherwise a transposed test instance would be unlearnable.
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


def sample_colors(num_examples=None) -> dict:
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"instance_plan": plan}


# ---------------------------------------------------------------- 2. generate

def generate(diff_lb, diff_ub, max_h, max_w, mirrored=None, **kw) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])

    cols = list(range(10))

    # dimensions of the un-mirrored (row-band) grid
    H = max_w if mirrored else max_h     # total number of rows
    W = max_h if mirrored else max_w     # number of columns

    H = max(4, min(30, H))
    W = max(4, min(30, W))

    maxnr = max(1, min(10, H // 2))
    nr = _unifint(diff_lb, diff_ub, (1, maxnr))
    w = _unifint(diff_lb, diff_ub, (4, W))

    scols = random.sample(cols, nr)
    sgs = [[[c] * w for _ in range(2)] for c in scols]     # each band: 2 rows

    max_exp = min(30 - nr, H - 2 * nr)
    if max_exp < 0:
        max_exp = 0
    numexp = _unifint(diff_lb, diff_ub, (0, max_exp))
    for _ in range(numexp):
        idx = random.randint(0, nr - 1)
        sgs[idx].append(list(sgs[idx][-1]))

    sgs2 = []
    for idx, col in enumerate(scols):
        sg = [list(r) for r in sgs[idx]]
        a, b = len(sg), len(sg[0])
        ub = max(0, (a * b) // 2 - 1)
        nnoise = _unifint(diff_lb, diff_ub, (0, ub))
        inds = [(i, j) for i in range(a) for j in range(b)]
        noise = random.sample(inds, nnoise)
        oc = [c for c in cols if c != col]
        for (i, j) in noise:
            sg[i][j] = random.choice(oc)
        # first and last row of every band keep at least half band-colour cells
        for idxx in (0, -1):
            while sum(1 for e in sg[idxx] if e == col) < b // 2:
                locs = [j for j, e in enumerate(sg[idxx]) if e != col]
                if not locs:
                    break
                sg[idxx][random.choice(locs)] = col
        sgs2.append(sg)

    gi = [row for sg in sgs2 for row in sg]
    go = [row for sg in sgs for row in sg]

    if mirrored:
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {"input": gi, "output": go}


# ---------------------------------------------------------------- 3. ops

def _segment_bands(G):
    """Segment a row-band grid into (start, end, colour) bands, using ONLY G.

    Band colours are sampled WITHOUT replacement and every band's first/last row
    holds at least half band-colour cells, while noise rows inside a band rarely
    concentrate one foreign colour that far.  So: a new band starts at row r when
    r's dominant colour is fresh (unused), reaches >= w//2 cells, and the current
    band is already at least 2 rows long (bands are built from >= 2 rows).
    """
    h, w = G.shape
    thr = max(2, w // 2)
    bands = []
    start = 0
    used = set()
    cur = _mostcommon(G[0].tolist())
    used.add(cur)
    r = 1
    while r < h:
        cnt = Counter(G[r].tolist())
        top, n = cnt.most_common(1)[0]
        if top != cur and top not in used and n >= thr and (r - start) >= 2 and (h - r) >= 2:
            block = G[start:r].flatten().tolist()
            bands.append((start, r - 1, _mostcommon(block)))
            start = r
            cur = top
            used.add(cur)
        r += 1
    block = G[start:h].flatten().tolist()
    bands.append((start, h - 1, _mostcommon(block)))
    return bands


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- orientation, measured from I only -------------------------------
    row_maj = [_mostcommon(I[r].tolist()) for r in range(hi)]
    col_maj = [_mostcommon(I[:, c].tolist()) for c in range(wi)]
    vertical = len(set(col_maj)) > len(set(row_maj))

    # work in "row-band" coordinates
    G = I.T if vertical else I
    bands = _segment_bands(G)

    # --- paint every band with its own majority colour, one op per band ---
    for (s, e, col) in bands:
        region = G[s:e + 1, :]
        if np.all(region == col):
            continue                     # band already uniform: nothing to do
        ops.append(int(col))
        if vertical:
            # exact full rectangle: columns s..e, all rows
            sels.append([0, s, hi - 1, e - s])
        else:
            # exact full rectangle: rows s..e, all columns
            sels.append([s, 0, e - s, wi - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e26a3af2"
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
                                f"for task e26a3af2"
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
                    f"Failed to build a complete episode for task e26a3af2 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e26a3af2-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
