from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np
import random


class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        dat = []
        num = 0

        num_samples = kwargs['num_samples']
        max_h, max_w = kwargs['max_grid_dim']
        num_examples = kwargs['num_examples']

        # randomly generate inputs
        while num < num_samples:
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []

            operations = []
            selections = []

            # Choose consistent background and object colors for this sample
            background_color = 0  # Fixed background as 0
            available_colors = [c for c in range(1, 10)]  # Use colors 1-9 for objects (9 colors)
            num_object_colors = random.randint(3, min(6, len(available_colors)))  # 3-6 colors, but not more than available
            object_colors = random.sample(available_colors, num_object_colors)

            # Generate training examples + 1 test case
            j = 0
            while (j < num_examples + 1):
                # 1. Generate grid dimensions
                grid_h = random.randint(4, max_h)
                grid_w = random.randint(4, max_w)

                # 2. Create input grid with random scattered objects
                input_grid = np.full((grid_h, grid_w), background_color, dtype=np.int8)

                # Fill with random pattern - one color per column
                for j_col in range(grid_w):
                    # Choose one color for this column
                    column_color = random.choice(object_colors)

                    # Place objects in this column with higher probability in upper rows
                    for i in range(grid_h):
                        # Higher probability in upper rows
                        prob = 0.3 * (1 - i / grid_h) + 0.1  # 0.4 at top, 0.1 at bottom
                        if random.random() < prob:
                            input_grid[i, j_col] = column_color

                # 3. Create output grid by applying gravity
                output_grid = np.full((grid_h, grid_w), background_color, dtype=np.int8)

                # Apply gravity to each column using count-based approach
                for col in range(grid_w):
                    # Count each color in this column
                    color_counts = {}
                    for row in range(grid_h):
                        color = input_grid[row, col]
                        if color != background_color:
                            color_counts[color] = color_counts.get(color, 0) + 1

                    # Fill from bottom with each color based on count
                    # Sort colors to ensure consistent ordering
                    current_row = grid_h - 1
                    for color in sorted(color_counts.keys()):
                        count = color_counts[color]
                        for _ in range(count):
                            output_grid[current_row, col] = color
                            current_row -= 1

                # 4. Add to examples or test case
                if j < num_examples:
                    ex_in.append(input_grid)
                    ex_out.append(output_grid)
                else:
                    pr_in.append(input_grid)
                    pr_out.append(output_grid)

                j += 1

            # 5. Record operations for gravity simulation
            # The output grid starts as a copy of input grid
            # We need to simulate moving objects down due to gravity

            # Create a working copy to track positions during operations
            # Use the test case input (last generated grid)
            working_grid = pr_in[0].copy()

            # Physical gravity simulation - process columns from left to right
            for col in range(grid_w):
                # Keep simulating until no more objects can fall in this column
                something_moved = True
                while something_moved:
                    something_moved = False

                    # Scan column from bottom to top to find blocks that can fall
                    for row in range(grid_h - 2, -1, -1):  # Start from second-to-bottom row
                        if working_grid[row, col] != background_color:
                            # Find connected block starting from this position
                            object_color = working_grid[row, col]
                            block_bottom = row
                            block_top = row

                            # Extend upward to find the top of the connected block
                            while (block_top - 1 >= 0 and
                                   working_grid[block_top - 1, col] == object_color):
                                block_top -= 1

                            block_height = block_bottom - block_top + 1

                            # Calculate how far this block can fall
                            max_fall_distance = 0
                            for check_row in range(block_bottom + 1, grid_h):
                                if working_grid[check_row, col] == background_color:
                                    max_fall_distance += 1
                                else:
                                    break  # Hit an obstacle

                            if max_fall_distance > 0:
                                # Move the block down by max_fall_distance steps (one MoveD per step)
                                # Selection updates each step to track the moving object
                                for step in range(max_fall_distance):
                                    operations.append(21)  # MoveD
                                    selections.append([block_top + step, col, block_height - 1, 0])

                                # Clear original positions
                                for clear_idx in range(block_height):
                                    working_grid[block_top + clear_idx, col] = background_color

                                # Place block at final position
                                for place_idx in range(block_height):
                                    working_grid[block_top + max_fall_distance + place_idx, col] = object_color

                                something_moved = True

            # Step 3: Submit
            operations.append(34)  # Submit
            selections.append([0, 0, grid_h-1, grid_w-1])

            # Only increment num when validation passes
            num += 1
            desc = {'id': f'1e0a9b12-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))

            # Generate random trajectories (wrong directions)
            # MoveU=20, MoveD=21, MoveL=22, MoveR=23
            wrong_directions = [20, 22, 23]  # U, L, R (not D which is correct)

            for rand_idx in range(1):
                # Branch from a random point in the operations
                if len(operations) > 2:
                    branch_idx = random.randint(1, len(operations) - 2)
                else:
                    branch_idx = 0

                rand_operations = operations[:branch_idx]
                rand_selections = selections[:branch_idx]

                # Create a working grid to track state during random moves
                # Apply the operations up to branch_idx to get the correct grid state
                rand_working_grid = pr_in[0].copy()
                for op_idx in range(branch_idx):
                    op = operations[op_idx]
                    sel = selections[op_idx]
                    if op == 21:  # MoveD
                        sel_row, sel_col, sel_h, sel_w = sel
                        block_height = sel_h + 1
                        # Move block down by 1
                        for h_idx in range(block_height - 1, -1, -1):
                            obj_color = rand_working_grid[sel_row + h_idx, sel_col]
                            rand_working_grid[sel_row + h_idx, sel_col] = background_color
                            rand_working_grid[sel_row + h_idx + 1, sel_col] = obj_color

                # Add random wrong moves with sanity check
                # Pick one random object and move it multiple times
                moves_added = 0
                attempts = 0
                current_row, current_col = None, None

                # Find initial object position
                while attempts < 30 and current_row is None:
                    attempts += 1
                    rand_row = random.randint(0, grid_h - 1)
                    rand_col = random.randint(0, grid_w - 1)
                    if rand_working_grid[rand_row, rand_col] != background_color:
                        current_row, current_col = rand_row, rand_col

                if current_row is None:
                    continue

                while moves_added < 6 and attempts < 60:
                    attempts += 1

                    # Choose a wrong direction (not MoveD)
                    direction = random.choice(wrong_directions)

                    # Check if movement is valid (no obstacle in that direction)
                    can_move = False
                    new_row, new_col = current_row, current_col
                    if direction == 20:  # MoveU
                        if current_row > 0 and rand_working_grid[current_row - 1, current_col] == background_color:
                            can_move = True
                            new_row = current_row - 1
                    elif direction == 22:  # MoveL
                        if current_col > 0 and rand_working_grid[current_row, current_col - 1] == background_color:
                            can_move = True
                            new_col = current_col - 1
                    elif direction == 23:  # MoveR
                        if current_col < grid_w - 1 and rand_working_grid[current_row, current_col + 1] == background_color:
                            can_move = True
                            new_col = current_col + 1

                    if can_move:
                        rand_operations.append(direction)
                        rand_selections.append([current_row, current_col, 0, 0])
                        moves_added += 1

                        # Update grid state after the move
                        obj_color = rand_working_grid[current_row, current_col]
                        rand_working_grid[current_row, current_col] = background_color
                        rand_working_grid[new_row, new_col] = obj_color

                        # Update current position to follow the moved object
                        current_row, current_col = new_row, new_col

                # Add submit
                rand_operations.append(34)
                rand_selections.append([0, 0, grid_h - 1, grid_w - 1])

                desc = {'id': f'1e0a9b12-random_{num}_{rand_idx}',
                        'selections': rand_selections,
                        'operations': rand_operations}
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))


        return dat
