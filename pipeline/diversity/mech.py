"""LLM 없이 기존 경로에서 새 경로를 만들 수 있는가.

세 가지 변환을 시도하고, 재생해서 답에 닿는 것만 센다. 닿지 않으면 그 변환은
이 경로에 쓸 수 없는 것이다 -- 판정은 추측이 아니라 ARCLE 재생이다.

  순열   서로 겹치지 않는 영역을 건드리는 이웃한 두 단계의 자리를 바꾼다.
         겹치면 결과가 달라질 수 있으므로 겹치는 쌍은 건드리지 않는다.
  분할   한 선택이 떨어진 덩어리 여러 개면, 덩어리마다 한 단계로 쪼갠다.
  병합   같은 op을 연달아 쓰는 이웃한 두 단계를 선택을 합쳐 한 단계로 만든다.
         Move류는 합치면 뜻이 달라지므로 Color/Flood 계열만 대상으로 한다.
"""
import sys, json, collections
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker/arc-agi-1")
OUT = Path("/hdd_data/yunho/cctmp/mech.json")
N = 3
MERGEABLE = {f"Color{i}" for i in range(10)} | {f"FloodFill{i}" for i in range(10)}

def mask(sel):
    return cm.solar_utils.to_sel_mask(sel, cm.MAX_GRID_DIM).astype(bool)

def cells(sel):
    rr, cc = np.where(mask(sel))
    return frozenset(zip(rr.tolist(), cc.tolist()))

def blobs(cs):
    """선택을 8-이웃 연결 덩어리로 쪼갠다."""
    todo, out = set(cs), []
    while todo:
        seed = todo.pop(); comp = {seed}; stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in todo:
                        todo.discard(p); comp.add(p); stack.append(p)
        out.append(sorted(comp))
    return out

def run(I, O, ops, sels):
    g = cm._run(np.asarray(I), np.asarray(O), ops, sels)
    return g is not None and g.shape == np.asarray(O).shape and bool((g == np.asarray(O)).all())

def variants(ops, sels):
    """(이름, ops, sels) 후보들."""
    out = []
    n = len(ops)
    for i in range(n - 2):          # Submit 앞까지
        a, b = cells(sels[i]), cells(sels[i + 1])
        if a and b and not (a & b):
            o = list(ops); s = list(sels)
            o[i], o[i+1] = o[i+1], o[i]; s[i], s[i+1] = s[i+1], s[i]
            out.append(("순열", o, s))
    for i in range(n - 1):
        cs = cells(sels[i])
        bl = blobs(cs)
        if len(bl) > 1:
            o = list(ops[:i]) + [ops[i]] * len(bl) + list(ops[i+1:])
            s = list(sels[:i]) + [{"cells": [list(p) for p in b]} for b in bl] \
                + list(sels[i+1:])
            out.append(("분할", o, s))
            break
    for i in range(n - 2):
        if ops[i] == ops[i+1] and cm.NAME.get(int(ops[i])) in MERGEABLE:
            u = sorted(cells(sels[i]) | cells(sels[i+1]))
            o = list(ops[:i]) + [ops[i]] + list(ops[i+2:])
            s = list(sels[:i]) + [{"cells": [list(p) for p in u]}] + list(sels[i+2:])
            out.append(("병합", o, s))
            break
    return out

tasks = sorted(p.name for p in ROOT.iterdir()
               if p.is_dir() and p.name != "__pycache__")
rows = json.loads(OUT.read_text()) if OUT.is_file() else []
done = {r["task"] for r in rows}
todo = [t for t in tasks if t not in done]
print(f"{len(todo)} to go", flush=True)

for k, t in enumerate(todo, 1):
    pairs = cm.draw(t, N, [0])
    if not pairs:
        rows.append({"task": t, "ok": {}, "tried": {}}); continue
    pairs = pairs[:N]
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    derive = cm.load_derive(ROOT / t / "grid_maker.py")
    ok = collections.Counter(); tried = collections.Counter()
    for i, (I, O) in enumerate(pairs):
        shown = eps[i % len(eps)][0] if eps else None
        g, ops, sels = cm.replay(derive, I, O, shown)
        if ops is None:
            continue
        for name, o, s in variants(ops, sels):
            tried[name] += 1
            try:
                if run(I, O, o, s):
                    ok[name] += 1
            except Exception:
                pass
    rows.append({"task": t, "ok": dict(ok), "tried": dict(tried)})
    tmp = OUT.with_suffix(".part"); tmp.write_text(json.dumps(rows)); tmp.replace(OUT)
    if k % 25 == 0:
        print(f"[{k}/{len(todo)}] {t}", flush=True)

agg_ok = collections.Counter(); agg_tr = collections.Counter()
for r in rows:
    agg_ok.update(r["ok"]); agg_tr.update(r["tried"])
print()
for name in ("순열", "분할", "병합"):
    o, tr = agg_ok[name], agg_tr[name]
    tasks_ok = sum(1 for r in rows if r["ok"].get(name))
    print(f"{name}  유효 {o}/{tr}  ({o/tr:.0%})" if tr else f"{name}  시도 없음",
          f" 적용되는 과제 {tasks_ok}")
