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


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    d = random.uniform(lb, ub)
    lo = a + int(round((b - a) * max(0.0, d - 0.25)))
    hi = a + int(round((b - a) * min(1.0, d + 0.25)))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def sample_colors(num_examples=None) -> dict:
    # background is hardcoded 0; the block colour is the only sampled role
    sqc = random.choice(range(1, 10))
    return {"sqc": sqc}


def generate(diff_lb, diff_ub, max_h, max_w, sqc=None, **kw) -> dict:
    bgc = 0
    if sqc is None:
        sqc = random.choice(range(1, 10))
    remcols = [c for c in range(1, 10) if c != sqc]
    k = random.choice([0, 1, 2, 3])  # number of CW quarter turns applied at the end
    hb, wb = (max_h, max_w) if k % 2 == 0 else (max_w, max_h)
    for _ in range(100):
        h = _unifint(diff_lb, diff_ub, (5, max(5, min(30, hb))))
        w = _unifint(diff_lb, diff_ub, (5, max(5, min(30, wb))))
        nsq = _unifint(diff_lb, diff_ub, (1, 8))
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
        if succ == 0:
            continue
        rows = set(np.nonzero(gir == sqc)[0].tolist())
        key = np.array([gil[i] if i in rows else bgc for i in range(h)])
        gi = np.concatenate([key[:, None], gir], axis=1)
        giro = gir.copy()
        for i in range(h):
            giro[i][gir[i] == sqc] = key[i]
        go = np.concatenate([key[:, None], giro], axis=1)
        gi = np.rot90(gi, -k)
        go = np.rot90(go, -k)
        return {"input": gi.tolist(), "output": go.tolist()}
    raise RuntimeError("generation failed")


def derive_operations(I, O, examples=None):
    """Each block cell takes the colour of the key-line entry on its own row
    (key = border column) or its own column (key = border row).  All stripes
    keyed by the same colour are recoloured in one stroke.  Only I is read."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    cnt = Counter(v for v in I.flatten().tolist() if v != 0)
    if cnt:
        # block colour: every keyed stripe holds >=2 block cells -> majority colour
        sqc = cnt.most_common(1)[0][0]
        keycells = [(r, c) for r in range(h) for c in range(w) if I[r, c] not in (0, sqc)]
        found = None
        for kind, idx in (('col', 0), ('col', w - 1), ('row', 0), ('row', h - 1)):
            line = I[:, idx] if kind == 'col' else I[idx, :]
            if sqc in line.tolist():
                continue
            pos = [(c if kind == 'col' else r) for r, c in keycells]
            if keycells and all(p == idx for p in pos):
                found = (kind, idx, line)
                break
        if found is not None:
            kind, idx, key = found
            keyl = key.tolist()
            order = []
            for v in keyl:
                if v != 0 and v not in order:
                    order.append(v)
            # one Color op per key colour: all block cells in the stripes keyed by it
            for col in order:
                stripes = [i for i, v in enumerate(keyl) if v == col]
                if kind == 'col':
                    cells = [(i, c) for i in stripes for c in range(w) if I[i, c] == sqc]
                else:
                    cells = [(r, i) for i in stripes for r in range(h) if I[r, i] == sqc]
                if cells:
                    ops.append(int(col))
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
