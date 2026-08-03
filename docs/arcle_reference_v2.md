# ARCLE O2ARCv2Env-v0 Reference (v2)

## Environment
Env id: `ARCLE/O2ARCv2Env-v0`  
Grid: working grid max {MAX_H}×{MAX_W}, padded with 0 (black). Colors: 0–9.

## State keys
| Key | Type | Description |
|-----|------|-------------|
| `grid` | ndarray (H,W) uint8 | Current working grid (padded to MAX_GRID_DIM) |
| `grid_dim` | (int, int) | Actual (height, width) of working grid |
| `input` | ndarray (H,W) uint8 | Input grid (padded, NEVER changes) |
| `input_dim` | (int, int) | Actual (height, width) of input grid |
| `clip` | ndarray (H,W) uint8 | Clipboard (padded) |
| `clip_dim` | (int, int) | Actual (height, width) of clipboard content |
| `selected` | ndarray (H,W) uint8 | Current selection mask (updated by ops) |

## Action space
```python
action = {
    "selection": np.ndarray,  # shape (MAX_H, MAX_W), dtype bool  1=selected 0=not
    "operation": int,         # integer 0–34
}
```

## Selection format (in grid_maker.py)
`[r, c, h, w]` → converted to mask via `sel_mask[r:r+h+1, c:c+w+1] = 1`
- `r` : top-left row (0-indexed)
- `c` : top-left col (0-indexed)
- `h` : height-1  (0 = single row)
- `w` : width-1   (0 = single col)

**Example**: select a 3×4 region from row 2, col 1 → `[2, 1, 2, 3]`

---

## Operations (0–34)

### 0–9 · Color\<n\>
Paint all **selected** cells to color `n`.  
Selection: any region. Works on any grid size.

### 10–19 · FloodFill\<n\>
BFS from seed, replace entire connected same-color region with color `n`.  
**CRITICAL**: the underlying selection mask must have exactly 1 True cell. In the
`[r, c, h, w]` bbox format, that means `h=0, w=0` — always use `[r, c, 0, 0]`. If not → NOOP.  
Seed = the single selected cell.

### 20–23 · Move\<U/D/R/L\>
Move selected object 1 cell. Uses **object-selection mode** (see below).  
20=Up, 21=Down, 22=Right, 23=Left.

### 24 · Rotate90 / 25 · Rotate270
Rotate selected object (see system prompt for CCW/CW mapping). Uses object-selection mode.
Square selections only — see "Non-square Rotate" below for non-square inputs.

### 26 · FlipH / 27 · FlipV
Flip selected object (see system prompt for left↔right/up↔down mapping).
Safe for any shape — no position change, so no square restriction.

### 28 · CopyI
Copy selected region from **INPUT** grid to clipboard.  
Bounds: `xmax < input_dim.h AND ymax < input_dim.w` (else NOOP).  
**Only nonzero cells** are copied to clipboard (zero stays zero in clip).  
Sets `clip_dim = (selection_h, selection_w)`.

### 29 · CopyO
Copy selected region from **current working grid** to clipboard.  
Bounds: within `grid_dim`. Only nonzero cells copied.

### 30 · Paste
Paste clipboard at top-left of selection.  
- Only the **top-left corner** of selection is used as paste origin (`xmin`, `ymin`).
- Pastes `clip[:clip_dim.h, :clip_dim.w]` starting at `(xmin, ymin)`.
- **Only nonzero** cells from clipboard overwrite grid (zero clipboard cells don't overwrite).
- Truncates at MAX_GRID_DIM ({MAX_H}×{MAX_W}).

So for paste at position `(r, c)`: use selection `[r, c, 0, 0]` (single cell is enough).

### 31 · CopyInput
Reset working grid to input grid.  
`grid[:] = input[:]`, `grid_dim = input_dim`.  
**Rarely needed**: at episode start the grid IS already the input. Only use if you need
to undo modifications and restart from input mid-sequence.

### 32 · ResetGrid
Set **all** working grid cells to 0. `grid_dim` unchanged. ⛔ Forbidden — see system prompt.

### 33 · CropGrid / ResizeGrid
Resize working canvas to selection bounding box.
- Computes bbox of selection → (xmin, xmax, ymin, ymax).
- Copies `grid[xmin:xmax+1, ymin:ymax+1]` to `grid[0:H, 0:W]`.
- **Only nonzero** cells within selection are preserved (transparent copy).
- Sets `grid_dim = (xmax-xmin+1, ymax-ymin+1)`.
- Clears rest of grid.

Use to **resize** working canvas (expand or shrink).

### 34 · Submit
Submit current working grid as answer. **Always the last operation.**

---

## Object-selection mode (ops 20–27)

Flip/Rotate/Move use an "object" abstraction:
1. On first call with a nonempty selection: the selected region is captured as an object
   (background preserved separately). Object buffer holds only nonzero cells of selection.
2. Operation applies to the object buffer.
3. Object is composited back onto background at its new position (transparent: zeros not written).

For a **whole-grid flip/rotate**: select the entire grid region `[0, 0, hi-1, wi-1]`.

---

## numpy ↔ ARCLE op mapping (for cross-referencing DSL/numpy names — semantics themselves are in the system prompt, not repeated here)

| numpy call | ARCLE op | DSL name | Shape restriction |
|------------|----------|----------|-------------------|
| `np.rot90(I, k=3)` | 25 | `rot90` (CW) | **square only** |
| `np.rot90(I, k=2)` | 26 then 27 | `rot180` | any shape |
| `np.rot90(I, k=1)` | 24 | `rot270` (CCW) | **square only** |
| `np.fliplr(I)` | 26 | `vmirror` | any shape |
| `np.flipud(I)` | 27 | `hmirror` | any shape |

---

## ⚠️ Non-square Rotate: resize first

**ARCLE Rotate (op24/op25) position math is wrong when the SELECTED REGION is non-square (h ≠ w).**

Root cause: `new_x = floor((h-w)/2) + pos_x` — negative for wide selections.

**Fix: make the selection square first via ResizeGrid, rotate, then CropGrid.**

```python
# Rotate CW (op25 = rot90) on non-square h×w input → output w×h
hi, wi = I.shape
sq = max(hi, wi)           # expand to square

ops, sels = [], []
# 1. Expand canvas to sq×sq (zero-pad; I stays at top-left)
ops.append(33); sels.append([0, 0, sq-1, sq-1])
# 2. Rotate CW on the full square selection (h==w → position correct)
ops.append(25); sels.append([0, 0, sq-1, sq-1])
# 3. After CW rotation of sq×sq, the valid output (w×h) lives at:
#    rows 0..wi-1, cols (sq-hi)..sq-1
#    CropGrid to extract it
ops.append(33); sels.append([0, sq-hi, wi-1, hi-1])
ops.append(34); sels.append([0, 0, wi-1, hi-1])
```

Derivation: CW rot90 of sq×sq: result[r,c] = padded[sq-1-c, r].
Input I occupies rows 0..hi-1, cols 0..wi-1.
Non-zero output: r in 0..wi-1, c in sq-hi..sq-1. Crop from (0, sq-hi).

For **CCW rotation (op24 = rot270)** on h×w → output w×h:
After CCW rotation of sq×sq: result[r,c] = padded[c, sq-1-r].
Non-zero output: r in sq-wi..sq-1, c in 0..hi-1. Crop from (sq-wi, 0).
```python
ops.append(33); sels.append([0, 0, sq-1, sq-1])    # expand
ops.append(24); sels.append([0, 0, sq-1, sq-1])    # CCW rotate
ops.append(33); sels.append([sq-wi, 0, wi-1, hi-1]) # crop
ops.append(34); sels.append([0, 0, wi-1, hi-1])
```

**rot180 (any shape, no resize needed)**: op26 then op27 directly on the h×w grid.

Always check the generator: if `w = h` is explicit → inputs are square → Rotate directly without resize.
If h and w are sampled independently → assume non-square possible → use resize+rotate+crop.

---

## Verified common patterns

Snippets below reference `bgc` for readability. `derive_operations(I, O)` only receives
I and O — bgc is NOT in scope there. Compute it locally before using these patterns:
`bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]`.

### Same-size geometric transform (no canvas resize needed)
```python
hi, wi = I.shape
sel = [0, 0, hi-1, wi-1]

# vmirror / left↔right (np.fliplr): O == np.fliplr(I)
# DSL: vmirror(I) — FlipH = op26
ops, sels = [26, 34], [sel, sel]

# hmirror / up↔down (np.flipud): O == np.flipud(I)
# DSL: hmirror(I) — FlipV = op27
ops, sels = [27, 34], [sel, sel]

# rot90 CW (np.rot90 k=3): O == np.rot90(I, 3)   ← SQUARE INPUTS ONLY
# DSL: rot90(I)
ops, sels = [25, 34], [sel, sel]

# rot270 / rot90 CCW (np.rot90 k=1): O == np.rot90(I, 1)  ← SQUARE INPUTS ONLY
# DSL: rot270(I)
ops, sels = [24, 34], [sel, sel]

# rot180 (fliplr then flipud): O == np.rot90(I, 2)  — ANY SHAPE
# DSL: rot180(I) = hmirror(vmirror(I)) = flipud(fliplr(I))
ops, sels = [26, 27, 34], [sel, sel, sel]
```

**Do NOT prepend CopyInput (op31)**: at episode start the grid is already the input.
op31 is redundant at the start and wastes a step. Only use op31 mid-sequence if you need
to revert to input after earlier modifications.

### Move object (translate by (dr, dc))
```python
# Move object at (obj_r, obj_c) of size (obj_h, obj_w) by (dr, dc)
# dr<0=up, dr>0=down; dc<0=left, dc>0=right
# op: 20=MoveU, 21=MoveD, 22=MoveR, 23=MoveL

ops, sels = [], []
r, c = obj_r, obj_c   # track current top-left
for _ in range(abs(dr)):
    move_op = 20 if dr < 0 else 21
    ops.append(move_op)
    sels.append([r, c, obj_h - 1, obj_w - 1])
    r += (-1 if dr < 0 else 1)
for _ in range(abs(dc)):
    move_op = 23 if dc < 0 else 22
    ops.append(move_op)
    sels.append([r, c, obj_h - 1, obj_w - 1])
    c += (-1 if dc < 0 else 1)
```

**CRITICAL**: Select the bbox at the object's CURRENT position before each step.

### Recolor via FloodFill
```python
ops.append(10 + new_color)   # FloodFill<new_color>
sels.append([r, c, 0, 0])    # single seed cell inside region
```

### Recolor via Color (rectangular region or individual cells)
```python
ops.append(n)                 # Color<n>
sels.append([r, c, h-1, w-1])
```

### Canvas expansion + copy-paste tiling
```python
# O = hconcat(I, transform(I)) — right half is some transform of I
hi, wi = I.shape
ho, wo = O.shape   # e.g. ho=hi, wo=2*wi

ops, sels = [], []
# 1. Expand canvas to output size (transparent copy of I to top-left)
ops.append(33); sels.append([0, 0, ho-1, wo-1])
# 2. Copy input nonzero cells to clipboard
ops.append(28); sels.append([0, 0, hi-1, wi-1])
# 3. If bgc != 0: restore left half (CropGrid's transparent copy left the bgc-colored
#    zero pixels wrong — only paste when there's actually a gap to fix)
if bgc != 0:
    ops.append(30); sels.append([0, 0, 0, 0])
# 4. Paste at right-half origin
ops.append(30); sels.append([0, wi, 0, 0])
# 5. Apply transform on right half (e.g. FlipH for vmirror)
ops.append(26); sels.append([0, wi, hi-1, wi-1])
# 6. Submit
ops.append(34); sels.append([0, 0, ho-1, wo-1])
```

**Why paste at (0,0)?** `CropGrid` (op33) is a *transparent* copy: zero cells in I are NOT
copied. If bgc≠0, the non-bgc zero pixels in the left half end up wrong, so the extra Paste
at (0,0) overwrites them with the clipboard's non-zero bgc cells.
**Skip this step when bgc==0** — the zero-padding from CropGrid already IS bgc, so the paste
would rewrite every cell in that half to the exact value it already has: a no-op per cell.

### General tiling: O = I repeated ny×nx times
```python
hi, wi = I.shape
ho, wo = O.shape   # ho = ny*hi, wo = nx*wi

ops, sels = [], []
ops.append(33);  sels.append([0, 0, ho-1, wo-1])         # expand canvas
ops.append(28);  sels.append([0, 0, hi-1, wi-1])         # CopyI
for ty in range(ny):
    for tx in range(nx):
        if ty == 0 and tx == 0: continue                  # top-left already there
        ops.append(30)
        sels.append([ty*hi, tx*wi, 0, 0])                 # Paste at tile origin
ops.append(34);  sels.append([0, 0, ho-1, wo-1])         # Submit
```

### Subgrid crop (extract sub-region)
```python
# O = I[r0:r0+ho, c0:c0+wo]
ops  = [33, 34]
sels = [[r0, c0, ho-1, wo-1], [0, 0, ho-1, wo-1]]
```

### ResizeGrid + fill background color
```python
# After ResizeGrid, canvas is expanded with zeros.
# If bgc != 0: the zero-padding is NOT bgc, so fill it explicitly before painting on top.
ops.append(33); sels.append([0, 0, ho-1, wo-1])   # resize
if bgc != 0:
    ops.append(bgc); sels.append([0, 0, ho-1, wo-1])  # fill bgc
# ... then paint non-bgc cells on top ...
```

**Skip the fill when bgc==0.** The zero-padding ResizeGrid produces already IS bgc in that case —
filling anyway repaints every already-correct cell to the value it already has (a no-op per cell,
and this region is usually the whole canvas, so it adds up fast). Only fill when bgc != 0, or when
this resize happens mid-sequence (not the first canvas-setup op) where earlier ops in the *same*
derivation may have left real stale nonzero content in that region — in that case check which cells
actually still differ from bgc rather than repainting the whole rectangle.

---

## Critical gotchas (implementation details not covered elsewhere)

| Issue | Detail |
|-------|---------|
| Paste bounds | Checked against `input.shape` = MAX_GRID_DIM = {MAX_H}×{MAX_W}. Fails if origin ≥ MAX_H or ≥ MAX_W. |
| Object ops position | After Rotate/Flip, object pos updates. Consecutive ops on same object don't need re-selection. |

---

## Episode design rules

- `sample_colors()` fixes ALL randomly-sampled colors ONCE per episode.
- `generate()` receives these as kwargs and must NOT re-sample them internally —
  this is what keeps all instances in an episode on the same color scheme.

---

## Template: derive_operations function

```python
import numpy as np

def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops  = []
    sels = []

    # ... task-specific logic ...

    ops.append(34)
    sels.append([0, 0, ho-1, wo-1])
    return ops, sels
```
