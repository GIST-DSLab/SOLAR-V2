"""세 지시로도 안 갈라진 41개에 네 번째 지시를 붙인다: 다르게 묶어라.

짧게(golf), 다시 뽑기(resample), 남의 풀이 참조(barc) -- 셋 다 이 과제들에서는
같은 궤적으로 수렴했다. 남은 자유도는 '무엇을 한 단위로 보는가'다. 한 물체씩
가던 것을 한 줄씩 가거나 격자 전체를 한 번에 가면, 같은 규칙이라도 단계의
개수와 영역의 묶음이 달라진다. 전에 다중경로를 '어휘 / 묶는 법 / 순서'로
갈랐을 때의 가운데 층을 직접 겨냥한다.
"""
import sys, json
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker/arc-agi-1")
TAIL = """

The rule it reads is right and the route reaches the answer. Nothing about either needs to change.
What must change is the unit of work. Look at what this route treats as one thing -- one object, one cell, one row, the whole grid -- and write the same rule against a different unit. If it goes object by object, do it a row at a time, or a colour at a time, or the whole grid at once. If it does the whole grid in one stroke, take it apart and do the parts. If it paints, see whether the same change is a move or a copy; if it moves, see whether it is a paint.
A longer route is fine. A route with more operations is fine. What is not fine is the same decomposition written again -- if you cannot see a second way to carve this task, say so plainly instead of renaming variables.
Two limits hold regardless.
Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations."""

tasks = json.load(open("/hdd_data/yunho/cctmp/solo41.json"))
out = {}
for t in tasks:
    src = (ROOT / t / "grid_maker.py").read_text()
    i = src.find("def derive_operations")
    j = src.find("\nclass GridMaker", i)
    code = src[i:j if j > 0 else None].rstrip()
    pairs = cm.draw(t, 2, [0])
    derive = cm.load_derive(ROOT / t / "grid_maker.py")
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    shown = eps[0][0] if eps else None
    note = ""
    if pairs:
        g, ops, sels = cm.replay(derive, pairs[0][0], pairs[0][1], shown)
        if ops:
            note = (f" On one instance it takes {len(ops)} operations: "
                    + " ".join(cm.NAME.get(int(o)) for o in ops[:12]) + ".")
    head = ("Task-specific quality requirement: this task's maker already solves it."
            + note + " Its derive_operations is:\n\n```python\n" + code + "\n```")
    out[t] = {"feedback": head + TAIL}

json.dump(out, open("/hdd_data/yunho/cctmp/group41_feedback.json", "w"), indent=1)
print("wrote", len(out))
