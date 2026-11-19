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
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # bgc is hardcoded to 0 by the generator. The rule does not depend on the
    # block colour (sqc), but fixing it for the whole episode keeps every
    # instance consistent and lets derive_operations read it from the
    # demonstrations.
    return {"sqc": random.choice(range(1, 10))}


def generate(diff_lb, diff_ub, max_h, max_w, sqc=None, **kw) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, int(round(a + (b - a) * d))))

    if sqc is None:
        sqc = random.choice(range(1, 10))
    bgc = 0
    cap = min(max_h, max_w)          # the rotation may swap h and w
    while True:
        h = unifint(diff_lb, diff_ub, (5, cap))
        w = unifint(diff_lb, diff_ub, (5, cap))
        remcols = [c for c in range(1, 10) if c != sqc]
        nsq = unifint(diff_lb, diff_ub, (1, 8))
        gir = np.zeros((h, w - 1), dtype=int)
        gil = [random.choice(remcols) for _ in range(h)]
        succ = fails = 0
        while succ < nsq and fails < nsq * 5:
            loci = random.randint(0, h - 3)
            locj = random.randint(0, w - 3)
            lock = random.randint(loci + 1, min(loci + max(1, 2 * h // 3), h - 1))
            locl = random.randint(locj + 1, min(locj + max(1, 2 * w // 3), w - 1))
            if lock < h and locl < w - 1:
                gir[loci:lock + 1, locj:locl + 1] = sqc
                succ += 1
            else:
                fails += 1
        rows = set(np.nonzero(gir == sqc)[0].tolist())
        if not rows:
            continue
        key = np.array([gil[i] if i in rows else bgc for i in range(h)])
        gi = np.concatenate([key[:, None], gir], axis=1)
        giro = gir.copy()
        for i in range(h):
            giro[i][gir[i] == sqc] = key[i]
        go = np.concatenate([key[:, None], giro], axis=1)
        k = random.choice([0, 1, 2, 3])
        gi = np.rot90(gi, -k)
        go = np.rot90(go, -k)
        return {"input": gi.tolist(), "output": go.tolist()}


def _sqc_from_examples(examples):
    """Block colour = the colour that gets repainted in the demonstrations."""
    if not examples:
        return None
    cnt = Counter()
    for a, b in examples:
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            continue
        cnt.update(a[a != b].tolist())
    return cnt.most_common(1)[0][0] if cnt else None


def _find_key(I, sqc_hint=None):
    """The key line is a border row or column. Every other non-zero cell has
    one block colour that never appears in the key. A key entry is non-zero
    exactly when its stripe holds block cells."""
    h, w = I.shape
    cands = [('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)]
    best = None
    present = set(I.flatten().tolist())
    for kind, idx in cands:
        if kind == 'col':
            key = I[:, idx]
            rest = np.delete(I, idx, axis=1)
            stripes = [rest[i, :] for i in range(h)]
        else:
            key = I[idx, :]
            rest = np.delete(I, idx, axis=0)
            stripes = [rest[:, i] for i in range(w)]
        rc = set(rest[rest != 0].tolist())
        if len(rc) != 1:
            continue
        sqc = rc.pop()
        if sqc_hint is not None and sqc != sqc_hint and sqc_hint in present:
            continue
        if sqc in set(key.tolist()):
            continue
        if not all((key[i] != 0) == bool((s == sqc).any()) for i, s in enumerate(stripes)):
            continue
        score = len(set(key[key != 0].tolist()))
        if best is None or score > best[0]:
            best = (score, kind, idx, key, sqc)
    return None if best is None else best[1:]


def derive_operations(I, O, examples=None):
    """The code-golf reading matches the maker's reading. After normalising the
    orientation, it walks each line along the key. On each line it takes the
    key colour and repaints that line's block cells with it. So the unit is one
    stripe per key entry, taken in key-line order, with the key colour held
    fixed along the stripe. Both readings agree, so the route stays the same:
    walk along the key line and issue one Color<key colour> op per stripe. The
    op selects exactly that stripe's block cells, read from I. O is never
    inspected."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    found = _find_key(I, _sqc_from_examples(examples))
    if found is not None:
        kind, key_idx, key, sqc = found
        for i in range(len(key)):
            col = int(key[i])
            if col == 0:
                continue
            if kind == 'col':
                cells = [(i, c) for c in range(w) if c != key_idx and I[i, c] == sqc]
            else:
                cells = [(r, i) for r in range(h) if r != key_idx and I[r, i] == sqc]
            if not cells:
                continue
            ops.append(col)               # Color<key colour> on this stripe's block cells
            sels.append(sel_of(cells))

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
