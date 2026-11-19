"""
ARC Task: 447fd412 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of

from collections import Counter, defaultdict


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, indic = sample(cols, 2)
    # The body of the template is REPLICATED with Copy/Paste, and ARCLE's clipboard
    # treats colour 0 as "nothing" (never copied, never pasted).  A body painted in
    # 0 could therefore not be duplicated, so keep the body colour non-zero.
    # (bgc / indic may be 0: a 0 background is simply never written, and a 0
    #  indicator is already present at the destination before the paste.)
    mainc = choice([c for c in cols if c not in (0, bgc, indic)])
    return {"bgc": bgc, "indic": indic, "mainc": mainc}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, indic: int, mainc: int) -> dict:
    for _attempt in range(40):
        h = unifint(diff_lb, diff_ub, (min(12, max_h), max_h))
        w = unifint(diff_lb, diff_ub, (min(12, max_w), max_w))
        oh = unifint(diff_lb, diff_ub, (1, 4))
        ow = unifint(diff_lb, diff_ub, (1, 4))
        if oh * ow < 3:
            if choice((True, False)):
                oh = unifint(diff_lb, diff_ub, (3, 4))
            else:
                ow = unifint(diff_lb, diff_ub, (3, 4))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (3, oh * ow))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        objt = totuple(obj)
        kk = len(obj)
        nindic = randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
        indicobj = set(sample(objt, nindic))
        mainobj = obj - indicobj
        obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        gi = canvas(bgc, (h, w))
        go = canvas(bgc, (h, w))
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        go = paint(go, plcd)
        # free cells: background, not touching the template and not inside the
        # template's bounding box (that box is copied wholesale later on, so it has
        # to stay clean)
        inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd)) - backdrop(toindices(plcd))
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
        tr = 0
        maxtr = 10 * noccs
        succ = 0
        fachi = min(5, max(1, min(h, w) // max(oh, ow) - 1))
        while succ < noccs and tr < maxtr:
            tr += 1
            fac = randint(1, fachi)
            outobj = upscale(obj, fac)
            inobj = sfilter(outobj, lambda cij: cij[0] == indic)
            hh, ww = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            inobjp = shift(inobj, loc)
            outobjp = shift(outobj, loc)
            box = frozenset((i, j) for i in range(loc[0], loc[0] + hh)
                            for j in range(loc[1], loc[1] + ww))
            # the WHOLE box has to be free: it is what gets copied / pasted
            if box.issubset(inds):
                succ += 1
                inds = (inds - box) - mapply(neighbors, box)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        if succ == 0:
            continue
        gia = np.array(gi, dtype=int)
        goa = np.array(go, dtype=int)
        # only keep a pair whose rule is recoverable from the input alone
        if Counter(gia.flatten().tolist()).most_common(1)[0][0] != bgc:
            continue
        info = _analyse(gia)
        if info is None:
            continue
        if not np.array_equal(_render(gia, info), goa):
            continue
        return {'input': gi, 'output': go}
    return {'input': gi, 'output': go}


# ── shared rule analysis (used by generate() for validation and by
#    derive_operations() to plan the trajectory) ───────────────────────────────

def _analyse(I):
    """Read the rule off the INPUT alone.

    Returns (bgc, indic, mainc, (tr0, tc0), oh, ow, ind_cells, body_cells,
             placements) where placements is a list of (r0, c0, fac): the
             top-left of the upscaled template for every indicator occurrence.
    """
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # connected components of the non-background cells (diagonal, multi-colour)
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] \
                                    and I[ny, nx] != bgc:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
                comps.append(cells)
    if not comps:
        return None

    # the template is the only component that uses two colours
    ti = max(range(len(comps)),
             key=lambda k: (len({int(I[r, c]) for r, c in comps[k]}), len(comps[k])))
    tmpl = comps[ti]
    tmpl_counts = Counter(int(I[r, c]) for r, c in tmpl)
    if len(tmpl_counts) < 2:
        return None
    other_counts = Counter(int(I[r, c]) for k, cs in enumerate(comps) if k != ti
                           for (r, c) in cs)
    if not other_counts:
        return None
    indic = other_counts.most_common(1)[0][0]           # the scattered colour
    body_cols = [col for col in tmpl_counts if col != indic]
    if len(body_cols) != 1:
        return None
    mainc = body_cols[0]

    tr0 = min(r for r, c in tmpl)
    tc0 = min(c for r, c in tmpl)
    norm = [(r - tr0, c - tc0, int(I[r, c])) for r, c in tmpl]
    ind_cells = sorted((a, b) for a, b, col in norm if col == indic)
    body_cells = sorted((a, b) for a, b, col in norm if col == mainc)
    if not ind_cells or not body_cells:
        return None
    oh = max(a for a, b, _ in norm) + 1
    ow = max(b for a, b, _ in norm) + 1

    # An occurrence may sit anywhere the indicator pattern (upscaled) appears with
    # free background around it — the task evaluates that on a canvas three times
    # the grid and crops back, so search on the same padded canvas.
    H, W = 3 * h, 3 * w
    P = np.full((H, W), bgc, dtype=int)
    P[h:2 * h, w:2 * w] = I
    comp_of = {}
    for k, cs in enumerate(comps):
        for (r, c) in cs:
            comp_of[(r + h, c + w)] = k
    comp_size = [len(cs) for cs in comps]
    anchor = ind_cells[0]
    ind_pos = [(r + h, c + w) for r in range(h) for c in range(w) if I[r, c] == indic]

    def matches(i, j, fac):
        for (a, b) in ind_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != indic:
                        return False
        for (a, b) in body_cells:
            for da in range(fac):
                for db in range(fac):
                    if P[i + a * fac + da, j + b * fac + db] != bgc:
                        return False
        return True

    placements = []
    tried = set()
    for fac in range(1, 6):
        need = len(ind_cells) * fac * fac
        for (pr, pc) in ind_pos:
            i = pr - anchor[0] * fac
            j = pc - anchor[1] * fac
            if (fac, i, j) in tried:
                continue
            tried.add((fac, i, j))
            if i < 0 or j < 0 or i + oh * fac > H or j + ow * fac > W:
                continue
            if not matches(i, j, fac):
                continue
            hit = set()
            for (a, b) in ind_cells + body_cells:
                for da in range(fac):
                    for db in range(fac):
                        k = comp_of.get((i + a * fac + da, j + b * fac + db))
                        if k is not None:
                            hit.add(k)
            if sum(comp_size[k] for k in hit) != need:
                continue
            placements.append((i - h, j - w, fac))
    placements.sort()
    return (bgc, indic, mainc, (tr0, tc0), oh, ow, ind_cells, body_cells, placements)


def _body_cells_at(r0, c0, fac, body_cells, h, w, clip=True):
    out = []
    for (a, b) in body_cells:
        for da in range(fac):
            for db in range(fac):
                r, c = r0 + a * fac + da, c0 + b * fac + db
                if not clip or (0 <= r < h and 0 <= c < w):
                    out.append((r, c))
    return sorted(set(out))


def _render(I, info):
    """The finished grid the rule asks for (used only to validate generate())."""
    bgc, indic, mainc, tl, oh, ow, ind_cells, body_cells, placements = info
    h, w = I.shape
    G = I.copy()
    for (r0, c0, fac) in placements:
        for (r, c) in _body_cells_at(r0, c0, fac, body_cells, h, w):
            G[r, c] = mainc
    return G


def derive_operations(I, O):
    """
    Rule: one two-coloured template sits in the grid.  Everywhere else only its
    INDICATOR subset appears, upscaled by some factor — each of those is a stamp of
    the template waiting to be completed.  The trajectory REPLICATES the template:
    copy it out of the input and paste it onto every same-size occurrence, and for
    each larger scale draw the enlarged template once and paste THAT onto the rest
    of that scale's occurrences.  Anything a paste cannot carry cleanly (a stamp
    hanging over the edge, a region already holding something else, a body colour
    of 0 — which ARCLE's clipboard treats as "nothing") is drawn instead.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    info = _analyse(I)
    if info is None:
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    bgc, indic, mainc, (tr0, tc0), oh, ow, ind_cells, body_cells, placements = info

    G = I.copy()                      # the grid as it stands after the ops so far

    by_fac = defaultdict(list)
    stray = []
    for (r0, c0, fac) in placements:
        if 0 <= r0 and 0 <= c0 and r0 + oh * fac <= h and c0 + ow * fac <= w:
            by_fac[fac].append((r0, c0))
        else:
            stray.append((r0, c0, fac))

    def body_at(r0, c0, fac):
        return _body_cells_at(r0, c0, fac, body_cells, h, w)

    def draw(grid, r0, c0, fac, apply_to):
        """Color op that completes one stamp (cells that are not mainc yet)."""
        cells = [rc for rc in body_at(r0, c0, fac) if grid[rc] != mainc]
        if not cells:
            return None
        for rc in cells:
            apply_to[rc] = mainc
        return cells

    def stamp(grid, clip, r0, c0, fac):
        """Paste `clip` at (r0,c0). Returns the resulting grid when the stamp lands
        cleanly — completing exactly this occurrence's body and disturbing nothing
        else — otherwise None."""
        ch, cw = clip.shape
        if r0 + ch > h or c0 + cw > w:
            return None
        after = grid.copy()
        np.copyto(after[r0:r0 + ch, c0:c0 + cw], clip, where=(clip > 0))
        want = grid.copy()
        for rc in body_at(r0, c0, fac):
            want[rc] = mainc
        if not np.array_equal(after, want) or np.array_equal(after, grid):
            return None
        return after

    def clean_box(grid, r0, c0, fac):
        """True when the stamp's bounding box holds nothing but that stamp."""
        own = set(body_at(r0, c0, fac))
        own |= {(r0 + a * fac + da, c0 + b * fac + db)
                for (a, b) in ind_cells for da in range(fac) for db in range(fac)}
        for r in range(r0, r0 + oh * fac):
            for c in range(c0, c0 + ow * fac):
                if (r, c) not in own and grid[r, c] != bgc:
                    return False
        return True

    def run_group(clip, spots, fac, copy_op, copy_sel):
        """Decide paste-vs-draw for every spot, then emit the group's ops."""
        scratch = G.copy()
        plan = []
        for (r0, c0) in spots:
            after = stamp(scratch, clip, r0, c0, fac)
            if after is not None:
                plan.append((r0, c0, True)); scratch = after
            else:
                plan.append((r0, c0, False))
                for rc in body_at(r0, c0, fac):
                    scratch[rc] = mainc
        if any(p[2] for p in plan):
            ops.append(copy_op); sels.append(copy_sel)
        for (r0, c0, pasted) in plan:
            if pasted:
                after = stamp(G, clip, r0, c0, fac)
                if after is not None:
                    ops.append(30); sels.append(sel_of([(r0, c0)]))
                    G[:, :] = after
                    continue
            cells = draw(G, r0, c0, fac, G)
            if cells:
                ops.append(int(mainc)); sels.append(sel_of(cells))

    # ── scale 1: the template itself is the stamp ────────────────────────────
    if by_fac.get(1):
        # bbox selection = exactly the template's bounding rectangle, its
        # background holes included, copied straight out of the input
        run_group(I[tr0:tr0 + oh, tc0:tc0 + ow].copy(), sorted(by_fac[1]), 1,
                  28, [tr0, tc0, oh - 1, ow - 1])

    # ── larger scales: ARCLE has no upscale op, so draw the enlarged template
    #    once and replicate THAT onto the other occurrences of the same scale ──
    for fac in sorted(k for k in by_fac if k > 1):
        spots = sorted(by_fac[fac])
        src = next((s for s in spots if clean_box(G, s[0], s[1], fac)), spots[0])
        cells = draw(G, src[0], src[1], fac, G)
        if cells:
            ops.append(int(mainc)); sels.append(sel_of(cells))
        rest = [s for s in spots if s != src]
        if rest:
            r0, c0 = src
            # bbox selection = exactly the enlarged template's bounding rectangle
            run_group(G[r0:r0 + oh * fac, c0:c0 + ow * fac].copy(), rest, fac,
                      29, [r0, c0, oh * fac - 1, ow * fac - 1])

    # ── a stamp hanging over the grid edge has no full copy to paste ─────────
    for (r0, c0, fac) in stray:
        cells = draw(G, r0, c0, fac, G)
        if cells:
            ops.append(int(mainc)); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 447fd412"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 447fd412"
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
                                f"for task 447fd412"
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
                    f"Failed to build a complete episode for task 447fd412 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"447fd412-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
