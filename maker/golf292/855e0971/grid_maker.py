"""
ARC Task: 855e0971 (RE-ARC) — LLM-generated grid_maker
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
    if b <= a:
        return a
    d = random.uniform(lb, ub)
    return max(a, min(b, a + int(round((b - a) * d + random.uniform(-0.5, 0.5)))))


def sample_colors(num_examples=None) -> dict:
    # dot colour is the only colour role the rule depends on; bar colours are free
    return {"dotc": random.randint(0, 9)}


def generate(diff_lb, diff_ub, max_h, max_w, dotc=None, mirrored=None) -> dict:
    if dotc is None:
        dotc = random.randint(0, 9)
    if mirrored is None:
        mirrored = random.choice((True, False))
    HH, WW = (max_w, max_h) if mirrored else (max_h, max_w)
    cols = list(range(10))
    nbarsd = _unifint(diff_lb, diff_ub, (1, 4))
    nbars = max(3, random.choice((nbarsd, 11 - nbarsd)))
    nbars = min(nbars, HH // 2)
    if nbars < 3:
        raise ValueError("grid too small")
    h = _unifint(diff_lb, diff_ub, (2 * nbars, min(30, HH)))
    w = _unifint(diff_lb, diff_ub, (3, min(30, WW)))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        barsizes[random.randint(0, nbars - 1)] += 1
    remcols = [c for c in cols if c != dotc]
    nloclbs = [random.choice((0, 1)) for _ in barsizes]
    if sum(nloclbs) < 2:
        a, b = random.sample(range(len(nloclbs)), 2)
        nloclbs[a] = 1
        nloclbs[b] = 1
    gi_rows, go_rows = [], []
    lastcol = -1
    for bs, lb in zip(barsizes, nloclbs):
        col = random.choice([c for c in remcols if c != lastcol])
        gim = np.full((bs, w), col, dtype=int)
        gom = gim.copy()
        nl = _unifint(diff_lb, diff_ub, (lb, w // 2))
        for jj in random.sample(range(w), nl):
            gim[random.randint(0, bs - 1), jj] = dotc
            gom[:, jj] = dotc
        lastcol = col
        gi_rows.append(gim)
        go_rows.append(gom)
    gi = np.vstack(gi_rows)
    go = np.vstack(go_rows)
    if mirrored:
        gi = gi.T
        go = go.T
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # orientation (verifier rule): bars are rows iff every row has < 3 colours
    horizontal = all(len(set(I[r].tolist())) < 3 for r in range(hi))
    G = I if horizontal else I.T
    gh, gw = G.shape

    # dot colour: the colour that grows in the demonstrations
    dotc = None
    if examples:
        votes = Counter()
        for ei, eo in examples:
            ci = Counter(np.asarray(ei).flatten().tolist())
            co = Counter(np.asarray(eo).flatten().tolist())
            for c in co:
                if co[c] > ci.get(c, 0):
                    votes[c] += 1
        if votes:
            dotc = votes.most_common(1)[0][0]
    if dotc is None or dotc not in set(G.flatten().tolist()):
        cands = []
        for c in set(G.flatten().tolist()):
            if all(len(set(G[r].tolist()) - {c}) == 1 for r in range(gh)):
                cands.append((int((G == c).sum()), c))
        dotc = min(cands)[1] if cands else Counter(G.flatten().tolist()).most_common()[-1][0]

    # bars: maximal runs of rows sharing the same bar colour (adjacent bars differ)
    barcol = []
    for r in range(gh):
        s = set(G[r].tolist()) - {dotc}
        barcol.append(s.pop() if s else None)
    bands = []
    r = 0
    while r < gh:
        r1 = r
        while r1 + 1 < gh and barcol[r1 + 1] == barcol[r]:
            r1 += 1
        bands.append((r, r1))
        r = r1 + 1

    ops, sels = [], []
    # one stroke per bar: every dot in the bar extends into a full line across that bar
    for (r0, r1) in bands:
        dcols = sorted({c for rr in range(r0, r1 + 1) for c in range(gw) if G[rr, c] == dotc})
        if not dcols:
            continue
        cells = [(rr, c) for rr in range(r0, r1 + 1) for c in dcols]
        if all(G[rr, c] == dotc for rr, c in cells):
            continue
        if not horizontal:
            cells = [(c, rr) for rr, c in cells]
        ops.append(int(dotc))
        sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])  # full-grid rectangle: submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 855e0971"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 855e0971"
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
                                f"for task 855e0971"
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
                    f"Failed to build a complete episode for task 855e0971 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"855e0971-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
