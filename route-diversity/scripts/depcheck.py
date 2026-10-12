"""다섯 과제의 golf 경로가 정답을 보고 그리는지 직접 확인한다.

세 가지를 묻는다.
  경로 변화   test의 답을 흐트러뜨렸을 때 (ops, sels)가 그대로인가.
              call_derive의 반환값 전체를 비교하므로 선택에 답을 넣은 경우도 잡힌다.
  남의 답      다른 인스턴스의 답을 건네면 그 답을 그려내는가.
  3인자        예시를 읽는 maker인가 (예시의 출력은 읽어도 되는 것이라 구분이 필요하다).
"""
import sys, inspect, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker")
TASKS = ["264363fd", "ce4f8723", "c3e719e8", "d0f5fe59", "c8cbb738"]
rng = np.random.default_rng(0)

print(f"{'task':10s} {'arm':8s} {'3인자':5s} {'경로바뀜':>9s} {'남의답그림':>11s}")
for t in TASKS:
    pairs = cm.draw(t, 10, [0])
    if not pairs:
        continue
    cp = cm.copy_pairs(pairs)
    for arm in ("arc-agi-1", "golf1"):
        p = ROOT / arm / t / "grid_maker.py"
        d = cm.load_derive(p)
        if d is None:
            continue
        try:
            three = len(inspect.signature(d).parameters) >= 3
        except Exception:
            three = False
        # 흐트러뜨렸을 때 경로가 바뀌는 비율 — 여러 번 흔든다
        tried = changed = 0
        for i, (I, O) in enumerate(pairs):
            shown = [(a.tolist(), b.tolist())
                     for j, (a, b) in enumerate(pairs) if j != i][:3]
            try:
                base = cm.call_derive(d, I.tolist(), O.tolist(), shown)
            except Exception:
                continue
            tried += 1
            for alt in cm.scrambles(O, np.random.default_rng(i)):
                try:
                    if cm.call_derive(d, I.tolist(), alt.tolist(), shown) != base:
                        changed += 1
                        break
                except Exception:
                    changed += 1
                    break
        drew = 0
        for I, P in cp:
            g, _, _ = cm.replay(d, I, P)
            if g is not None and g.shape == P.shape and bool((g == P).all()):
                drew += 1
        print(f"{t:10s} {arm:8s} {'예' if three else '아니오':5s} "
              f"{changed:3d}/{tried:<3d}      {drew:3d}/{len(cp):<3d}")
