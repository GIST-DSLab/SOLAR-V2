# 각 갈래에 실제로 넣은 지시문

과제마다 앞부분(현재 코드·op 수)은 다르고, 아래는 공통으로 붙은 지시다.
전문은 같은 폴더의 JSON에 과제별로 들어 있다.

## barc_feedback  (160 tasks)

> That is one person's reading of this task, written in their own vocabulary rather than ARCLE's. Where it carves the problem differently from the maker above -- different objects, a different order, a different thing held fixed -- follow the reading in the reference, not the one already here. Where the two agree, there is nothing to change and you should say so.
> The point is not a shorter route or a more accurate one. It is a different decomposition of the same rule, so write the operations that reading implies even when they are more numerous.
> Two limits hold regardless.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection, and the route then teaches nothing.
> And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.

## golf2_feedback  (25 tasks)

> Write a derive_operations that reaches the same answer in fewer operations. Fewer is better, always. Two operations beats ten; one beats two. Use whatever ARCLE allows to get there.
> The only hard requirement is that the trajectory replays to the correct output grid.

## golf84_feedback  (84 tasks)

> Solve the same task in as few operations as you can. Not by doing less, but by doing the same work in larger strokes: a whole region at once instead of cell by cell, one copy instead of repeating a shape, resizing the canvas instead of painting a border.
> Two limits, and the shortest route that respects them is the one to write.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection that is a list of cells you assembled by looking at the answer is not a shorter solution; it is the answer written into the selection, and the route then teaches nothing.
> And the operations must still be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.
> If the task genuinely needs one operation per cell, say so by writing that; a rule that is 'colour these particular cells' does not get shorter by force.

## golf_feedback  (25 tasks)

> Solve the same task in as few operations as you can. Not by doing less, but by doing the same work in larger strokes: a whole region at once instead of cell by cell, one copy instead of repeating a shape, resizing the canvas instead of painting a border.
> Two limits, and the shortest route that respects them is the one to write.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection that is a list of cells you assembled by looking at the answer is not a shorter solution; it is the answer written into the selection, and the route then teaches nothing.
> And the operations must still be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.
> If the task genuinely needs one operation per cell, say so by writing that; a rule that is 'colour these particular cells' does not get shorter by force.

## group105_feedback  (105 tasks)

> The rule it reads is right and the route reaches the answer. Nothing about either needs to change.
> What must change is the unit of work. Look at what this route treats as one thing -- one object, one cell, one row, the whole grid -- and write the same rule against a different unit. If it goes object by object, do it a row at a time, or a colour at a time, or the whole grid at once. If it does the whole grid in one stroke, take it apart and do the parts. If it paints, see whether the same change is a move or a copy; if it moves, see whether it is a paint.
> A longer route is fine. A route with more operations is fine. What is not fine is the same decomposition written again -- if you cannot see a second way to carve this task, say so plainly instead of renaming variables.
> Two limits hold regardless.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
> And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.

## group3_feedback  (254 tasks)

> The rule it reads is right and the route reaches the answer. Nothing about either needs to change.
> What must change is the unit of work. Look at what this route treats as one thing -- one object, one cell, one row, the whole grid -- and write the same rule against a different unit. If it goes object by object, do it a row at a time, or a colour at a time, or the whole grid at once. If it does the whole grid in one stroke, take it apart and do the parts. If it paints, see whether the same change is a move or a copy; if it moves, see whether it is a paint.
> A longer route is fine. A route with more operations is fine. What is not fine is the same decomposition written again -- if you cannot see a second way to carve this task, say so plainly instead of renaming variables.
> Two limits hold regardless.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
> And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.

## group41_feedback  (41 tasks)

> The rule it reads is right and the route reaches the answer. Nothing about either needs to change.
> What must change is the unit of work. Look at what this route treats as one thing -- one object, one cell, one row, the whole grid -- and write the same rule against a different unit. If it goes object by object, do it a row at a time, or a colour at a time, or the whole grid at once. If it does the whole grid in one stroke, take it apart and do the parts. If it paints, see whether the same change is a move or a copy; if it moves, see whether it is a paint.
> A longer route is fine. A route with more operations is fine. What is not fine is the same decomposition written again -- if you cannot see a second way to carve this task, say so plainly instead of renaming variables.
> Two limits hold regardless.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
> And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.

## retry10_feedback  (10 tasks)

> On this instance it is wrong.
> INPUT:
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 6 6 6 6 6 6 6 1 1 1 1 1 4 4 4 4 4 4 4 8 8 8 8 8 8 4
> 6 6 6 6 6 6 6 1 1 1 1 1 4 4 4 4 4 4 4 8 8 8 8 8 8 4
> 6 6 6 6 6 6 6 1 1 1 1 1 4 4 4 4 4 4 4 8 8 8 8 8 8 4
> 4 4 4 4 4 4 4 1 1 1 1 1 4 4 4 4 4 4 4 8 8 8 8 8 8 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 4 8 8 8 8 8 8 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 2 2 2 2 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 5 5 5 5 5 5 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 4 4 4 4 4 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 0 0 0 0 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 0 0 0 0 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 0 0 0 0 4 4 2 2 2 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> CORRECT OUTPUT:
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 4 4 4 4 4 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4
> 6 6 6 6 6 6 6 1 1 1 1 1 4 4 4 4 4 4 4 8 8 8 8 8 8 4
> 6 4 4 4 4 4 6 1 4 4 4 1 4 4 4 4 4 4 4 8 4 4 4 4 8 4
> 6 6 6 6 6 6 6 1 4 4 4 1 4 4 4 4 4 4 4 8 4 4 4 4 8 4
> 4 4 4 4 4 4 4 1 1 1 1 1 4 4 4 4 4 4 4 8 4 4 4 4 8 4
> 4 4 4 4 4 4 4 4 4 4 5 5 5 5 5 5 4 4 4 8 8 8 8 8 8 4
> 4 4 4 4 4 4 4 4 4 4 5 4 4 4 4 5 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 4 4 4 4 5 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 4 4 4 4 5 4 4 2 2 2 2 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 5 4 4 4 4 5 4 4 2 4 4 2 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 5 5 5 5 5 5 4 4 2 4 4 2 4 4 4 4
> 4 4 9 4 4 4 4 4 9 4 4 4 4 4 4 4 4 4 2 4 4 2 4 4 4 4
> 4 4 9 4 4 4 4 4 9 4 4 0 0 0 0 0 4 4 2 4 4 2 4 4 4 4
> 4 4 9 4 4 4 4 4 9 4 4 0 4 4 4 0 4 4 2 4 4 2 4 4 4 4
> 4 4 9 4 4 4 4 4 9 4 4 0 4 4 4 0 4 4 2 2 2 2 4 4 4 4
> 4 4 9 4 4 4 4 4 9 4 4 0 4 4 4 0 4 4 4 4 4 4 4 4 4 4
> 4 4 9 9 9 9 9 9 9 4 4 0 4 4 4 0 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 0 4 4 4 0 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 0 0 0 0 0 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 4 4 4 4 4 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 4 4 4 4 4 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 7 7 7 7 7 7 7 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> WHAT YOUR OPERATIONS PRODUCED:
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 4 6 6 6 6 6 3 3 3 3 3 3 3 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 3 6 6 6 6 6 3 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 3 3 3 3 3 3 3 6 6 6 6 6 6 6 6 6 6 6 6 4
> 6 6 6 6 6 6 6 1 1 1 1 1 6 6 6 6 6 6 6 8 8 8 8 8 8 4
> 6 6 6 6 6 6 6 1 6 6 6 1 6 6 6 6 6 6 6 8 6 6 6 6 8 4
> 6 6 6 6 6 6 6 1 6 6 6 1 6 6 6 6 6 6 6 8 6 6 6 6 8 4
> 4 6 6 6 6 6 6 1 1 1 1 1 6 6 6 6 6 6 6 8 6 6 6 6 8 4
> 4 6 6 6 6 6 6 6 6 6 5 5 5 5 5 5 6 6 6 8 8 8 8 8 8 4
> 4 6 6 6 6 6 6 6 6 6 5 6 6 6 6 5 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 5 6 6 6 6 5 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 5 6 6 6 6 5 6 6 2 2 2 2 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 5 6 6 6 6 5 6 6 2 6 6 2 6 6 6 4
> 4 6 9 9 9 9 9 9 9 6 5 5 5 5 5 5 6 6 2 6 6 2 6 6 6 4
> 4 6 9 6 6 6 6 6 9 6 6 6 6 6 6 6 6 6 2 6 6 2 6 6 6 4
> 4 6 9 6 6 6 6 6 9 6 6 0 0 0 0 0 6 6 2 6 6 2 6 6 6 4
> 4 6 9 6 6 6 6 6 9 6 6 0 6 6 6 0 6 6 2 6 6 2 6 6 6 4
> 4 6 9 6 6 6 6 6 9 6 6 0 6 6 6 0 6 6 2 2 2 2 6 6 6 4
> 4 6 9 6 6 6 6 6 9 6 6 0 6 6 6 0 6 6 6 6 6 6 6 6 6 4
> 4 6 9 9 9 9 9 9 9 6 6 0 6 6 6 0 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 6 0 6 6 6 0 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 6 0 0 0 0 0 6 6 6 6 6 6 6 6 6 4
> 4 6 7 7 7 7 7 7 7 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 7 6 6 6 6 6 7 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 7 6 6 6 6 6 7 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 7 7 7 7 7 7 7 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 4
> 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4
> 
> Keep the new decomposition and make it cover this instance too. If the decomposition cannot be made to cover it, go back to the one that works and say why the other reading does not hold.


## cg_* (code-golf 참조, 6개 arm)

과제마다 앞부분에 ①현재 maker의 derive_operations 전문 ②그 팀의 제출 코드가
붙고, 아래가 공통 꼬리다. arm마다 **한 팀의 코드만** 넣는다.

- `cg_cgi` (400 tasks) — `Seek64/NeurIPS-Code-Golf-2025`
- `cg_oxjam` (398 tasks) — `xsot/google-code-golf-2025`
- `cg_camjam` (396 tasks) — `intgrah/golf`
- `cg_aggregate` (399 tasks) — `arc-code-golf/solutions`
- `cg_itertools` (369 tasks) — `Natanaelel/GoogleCodeGolf2025`
- `cg_lv1dev` (368 tasks) — `lv1-dev/google-code-golf-2025`

> That is how one code-golf team solved the same task. It is written to be short in bytes, not to be read, so its shape is not the point -- what is worth taking is the reading of the task underneath it.
> Where that reading carves the problem differently from the maker above -- a different thing treated as one unit, a different order, a different quantity held fixed -- write the operations that reading implies. Where the two agree there is nothing to change and you should say so.
> Do not try to make your route short. The reference is short because bytes were the objective there; here they are not. A longer route that follows a different reading is the wanted answer.
> Two limits hold regardless.
> Every selection must be something you can name from the input before you know the answer -- a rectangle, a connected object, all cells of one colour, a row, a column, the cells a copy lands on. A selection assembled by looking at the answer is the answer written into the selection.
> And the operations must be derivable from the input grid and the worked examples alone: handed a different answer grid, this function must return the same operations.
