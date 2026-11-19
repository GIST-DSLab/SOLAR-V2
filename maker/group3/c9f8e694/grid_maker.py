"""
ARC Task: c9f8e694 (RE-ARC) — LLM-generated grid_maker
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
    # background is hard-coded to 0 in the generator; the block colour is sampled once
    sqc = random.choice(list(range(1, 10)))
    return {"sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, sqc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    bgc = 0
    if sqc is None:
        sqc = random.choice(list(range(1, 10)))
    remcols = [c for c in range(1, 10) if c != sqc]
    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    h = max(5, min(h, max_h))
    w = max(5, min(w, max_w))
    nsq = unifint(diff_lb, diff_ub, (1, 8))
    nsq = max(1, nsq)

    gir = np.zeros((h, w - 1), dtype=int)
    gil = [random.choice(remcols) for _ in range(h)]
    succ = 0
    fails = 0
    maxfails = nsq * 5
    while succ < nsq and fails < maxfails:
        loci = random.randint(0, h - 3)
        locj = random.randint(0, w - 3)
        lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
        locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
        if lock < h and locl < w - 1:
            gir[loci:lock + 1, locj:locl + 1] = sqc
            succ += 1
        else:
            fails += 1
    if not (gir == sqc).any():
        gir[0:2, 0:2] = sqc
    rows_with = set(np.where((gir == sqc).any(axis=1))[0].tolist())
    gil = np.array([[gil[i] if i in rows_with else bgc] for i in range(h)], dtype=int)
    giro = gir.copy()
    for i in range(h):
        giro[i][gir[i] == sqc] = gil[i, 0]
    gi = np.hstack([gil, gir])
    go = np.hstack([gil, giro])
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k=-k)
    go = np.rot90(go, k=-k)
    return {"input": gi.tolist(), "output": go.tolist()}


def _find_key(I):
    """Return (kind, idx, key, sqc): kind 'col' -> key is column idx (entries per row),
    'row' -> key is row idx (entries per column)."""
    h, w = I.shape
    cands = [('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)]
    for kind, idx in cands:
        if kind == 'col':
            key = I[:, idx]
            rest = np.delete(I, idx, axis=1)
        else:
            key = I[idx, :]
            rest = np.delete(I, idx, axis=0)
        rc = set(np.unique(rest).tolist()) - {0}
        if len(rc) != 1:
            continue
        sqc = rc.pop()
        if sqc in set(key.tolist()):
            continue
        has = (rest == sqc).any(axis=1) if kind == 'col' else (rest == sqc).any(axis=0)
        if all((key[i] != 0) == bool(has[i]) for i in range(len(key))):
            return kind, idx, key, sqc
    return None


def derive_operations(I, O, examples=None):
    """Same rule (each block cell takes the key entry of its own stripe), carved by
    COLOUR rather than by stripe: one Color<k> op paints every block cell lying on any
    stripe whose key entry is k, across the whole grid at once.  Measured from I only."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    found = _find_key(I)
    if found is not None:
        kind, kidx, key, sqc = found
        order = []
        for v in key.tolist():
            if v != 0 and v not in order:
                order.append(v)
        for k in order:
            stripes = [i for i in range(len(key)) if key[i] == k]
            cells = []
            for i in stripes:
                if kind == 'col':
                    cells += [(i, c) for c in range(w) if c != kidx and I[i, c] == sqc]
                else:
                    cells += [(r, i) for r in range(h) if r != kidx and I[r, i] == sqc]
            if not cells:
                continue
            ops.append(int(k))          # Color<k> on all block cells of k-keyed stripes
            sels.append(sel_of(sorted(cells)))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task c9f8e694"
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
                                f"for task c9f8e694"
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
                    f"Failed to build a complete episode for task c9f8e694 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"c9f8e694-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
