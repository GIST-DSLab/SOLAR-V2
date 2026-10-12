"""경로가 한 가지뿐인 84개 과제에 codegolf 지시를 붙인다.

golf1을 만든 그 문장을 그대로 쓴다. 25개 표본에서 23개가 다른 데서 안 나온
경로를 냈고, 답을 읽은 경우는 0이었다(가드를 뺀 golf2는 2개가 읽었다). 목표가
'짧게'에서 '다르게'로 바뀌었지만 지시는 바꾸지 않는다 -- 바꾸면 그 23/25가
무엇의 성과였는지 알 수 없게 된다.
"""
import sys, json, inspect
sys.path.insert(0, "/hdd_data/yunho/solar-traj/pipeline")
import numpy as np
import choose_maker as cm
from pathlib import Path

ROOT = Path("/hdd_data/yunho/solar-traj/maker/arc-agi-1")
TAIL = """

Solve the same task in as few operations as you can. Not by doing less, but by doing the same work in larger strokes: a whole region at once instead of cell by cell, one copy instead of repeating a shape, resizing the canvas instead of painting a border.
Two limits, and the shortest route that respects them is the one to write.
Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection that is a list of cells you assembled by looking at the answer is not a shorter solution; it is the answer written into the selection, and the route then teaches nothing.
And the operations must still be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.
If the task genuinely needs one operation per cell, say so by writing that; a rule that is 'colour these particular cells' does not get shorter by force."""

tasks = json.load(open("/hdd_data/yunho/cctmp/solo84.json"))
out, skipped = {}, []
for t in tasks:
    src = (ROOT / t / "grid_maker.py").read_text()
    i = src.find("def derive_operations")
    if i < 0:
        skipped.append((t, "no derive_operations")); continue
    j = src.find("\nclass GridMaker", i)
    code = src[i:j if j > 0 else None].rstrip()
    pairs = cm.draw(t, 3, [0])
    if not pairs:
        skipped.append((t, "no instances")); continue
    eps = cm.episodes_from_draw("/hdd_data/yunho/ARC_rearc_draw20", t) or []
    derive = cm.load_derive(ROOT / t / "grid_maker.py")
    shown = eps[0][0] if eps else None
    I, O = pairs[0]
    g, ops, sels = cm.replay(derive, I, O, shown)
    if ops is None:
        skipped.append((t, "no route")); continue
    head = (f"Task-specific quality requirement: this task's maker already solves "
            f"it, and the rule it reads is right. On one instance it takes "
            f"{len(ops)} operations, beginning "
            + " ".join(cm.NAME.get(int(o)) for o in ops[:10])
            + ". Its derive_operations is:\n\n")
    out[t] = {"feedback": head + "```python\n" + code + "\n```" + TAIL}

json.dump(out, open("/hdd_data/yunho/cctmp/golf84_feedback.json", "w"), indent=1)
print("wrote", len(out), "tasks; skipped", len(skipped), skipped[:5])
