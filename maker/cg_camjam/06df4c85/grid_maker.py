"""
ARC Task: 06df4c85 (RE-ARC) — LLM-generated grid_maker
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
    cols = list(range(10))
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, max(3, (max_h + 1) // (oh + 1))))
    numw = unifint(diff_lb, diff_ub, (3, max(3, (max_w + 1) // (ow + 1))))
    remcols = [c for c in range(10) if c not in (bgc, linc)]
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = random.sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = np.full((fullh, fullw), linc, dtype=int)
    for a in range(numh):
        for b in range(numw):
            r0, c0 = a * (oh + 1), b * (ow + 1)
            gi[r0:r0 + oh, c0:c0 + ow] = bgc
    go = gi.copy()

    def free_blocks(g):
        res = set()
        for a in range(numh):
            for b in range(numw):
                r0, c0 = a * (oh + 1), b * (ow + 1)
                if np.all(g[r0:r0 + oh, c0:c0 + ow] == bgc):
                    res.add((r0, c0))
        return res

    def fillb(g, loc, col):
        g[loc[0]:loc[0] + oh, loc[1]:loc[1] + ow] = col

    for col in ccols:
        inds = free_blocks(go)
        if len(inds) == 0:
            break
        loc = random.choice(sorted(inds))
        narms = random.randint(1, 4)
        armdirs = random.sample([(-1, 0), (1, 0), (0, -1), (0, 1)], narms)
        for (x, y) in armdirs:
            arm = []
            for k in range(1, max(numh, numw)):
                nl = (loc[0] + k * x * (oh + 1), loc[1] + k * y * (ow + 1))
                if nl not in inds:
                    break
                arm.append(nl)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            fillb(gi, arm[aidx], col)
            for e in arm[:aidx + 1]:
                fillb(go, e, col)
        fillb(gi, loc, col)
        fillb(go, loc, col)
    return {"input": gi.tolist(), "output": go.tolist()}


def _lattice(G):
    G = np.asarray(G, dtype=int)
    h, w = G.shape
    linc = None
    for r in range(h):
        if len(set(G[r].tolist())) == 1:
            linc = int(G[r, 0])
            break
    line_rows = [r for r in range(h) if bool(np.all(G[r] == linc))]
    line_cols = [c for c in range(w) if bool(np.all(G[:, c] == linc))]
    oh, ow = line_rows[0], line_cols[0]
    rs = list(range(0, h, oh + 1))
    cs = list(range(0, w, ow + 1))
    B = [[int(G[r, c]) for c in cs] for r in rs]
    return oh, ow, rs, cs, B


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    oh, ow, rs, cs, B = _lattice(I)
    nh, nw = len(rs), len(cs)

    # background block colour: the episode-wide majority block colour
    cnt = Counter(B[a][b] for a in range(nh) for b in range(nw))
    for ex in (examples or []):
        try:
            _, _, _, _, Be = _lattice(ex[0])
            cnt.update(v for row in Be for v in row)
        except Exception:
            pass
    bgc = cnt.most_common(1)[0][0]

    def block_cells(a, b):
        r0, c0 = rs[a], cs[b]
        return [(r, c) for r in range(r0, r0 + oh) for c in range(c0, c0 + ow)]

    ops, sels = [], []

    # Reading: along each lattice LINE (row, then column), a colour appearing on
    # both sides of a stretch of background blocks fills that whole stretch.
    # One op per bridged segment = the line of blocks strictly between the pair.
    lines = [[(a, b) for b in range(nw)] for a in range(nh)] + \
            [[(a, b) for a in range(nh)] for b in range(nw)]
    for line in lines:
        vals = [B[a][b] for (a, b) in line]
        pos = [i for i, v in enumerate(vals) if v != bgc]
        for i, j in zip(pos, pos[1:]):
            if vals[i] != vals[j] or j - i < 2:
                continue
            col = vals[i]
            seg = line[i + 1:j]
            if not all(vals[k] == bgc for k in range(i + 1, j)):
                continue
            cells = []
            for (a, b) in seg:
                cells.extend(block_cells(a, b))
            ops.append(col)
            sels.append(sel_of(cells))
            for (a, b) in seg:
                B[a][b] = col  # visible grid now holds it

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 06df4c85"
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
                                f"for task 06df4c85"
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
                    f"Failed to build a complete episode for task 06df4c85 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"06df4c85-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
