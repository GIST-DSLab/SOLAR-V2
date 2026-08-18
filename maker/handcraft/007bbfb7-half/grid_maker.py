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
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []

            operations = []
            selections = []
            background_color = np.random.randint(1, 10) 
            max_input_size = min(int(np.sqrt(max_h)), int(np.sqrt(max_w)))
            choices = list(range(2, max_input_size + 1))
            grid_sizes = [choice * choice for choice in choices]
            total_size = sum(grid_sizes)
            probabilities = [size / total_size for size in grid_sizes]
            input_h = np.random.choice(choices, p=probabilities)
            input_w = input_h
            
            # Generate training examples + 1 test case
            j = 0
            while (j < num_examples + 1):

                object_color = background_color
                while object_color == background_color:
                    object_color = np.random.randint(1, 10)  # Randomly select an object color

                # 2. Create input grid with random pattern
                # Ensure input grid is not all the same color
                while True:
                    input_grid = np.random.choice([background_color,object_color], size=[input_h, input_w])
                    # Check if grid has both colors
                    unique_colors = np.unique(input_grid)
                    if len(unique_colors) > 1:
                        break

                # 3. Create output grid (upscaled by factor equal to input size)
                scale_factor = input_h
                output_h = input_h * scale_factor
                output_w = input_w * scale_factor
                
                # Ensure output doesn't exceed max dimensions
                if output_h > max_h or output_w > max_w:
                    scale_factor = min(max_h // input_h, max_w // input_w)
                    output_h = input_h * scale_factor
                    output_w = input_w * scale_factor
                
                # 4. Create upscaled output
                output_grid = np.full((output_h, output_w), background_color, dtype=np.uint8)
                
                # Only place patterns where object color exists in input
                for input_i in range(input_h):
                    for input_j in range(input_w):
                        if input_grid[input_i, input_j] == object_color:
                            # Place the entire input pattern at the corresponding output position
                            paste_start_row = input_i * scale_factor
                            paste_start_col = input_j * scale_factor
                            
                            for pattern_i in range(input_h):
                                for pattern_j in range(input_w):
                                    output_i = paste_start_row + pattern_i
                                    output_j = paste_start_col + pattern_j
                                    if output_i < output_h and output_j < output_w:
                                        output_grid[output_i, output_j] = input_grid[pattern_i, pattern_j]
                
                # 5. Add to examples or test case
                if j < num_examples:
                    ex_in.append(input_grid)
                    ex_out.append(output_grid)
                else:
                    pr_in.append(input_grid)
                    pr_out.append(output_grid)
                
                j += 1

            # 6. Record operations using Copy-Paste approach
            # Step 1: Copy the entire input pattern
            operations.append(33)  # ResizeGrid
            selections.append([0, 0, output_h-1, output_w-1])  # New grid size
            
             # Step 2: Copy entire input pattern once
            operations.append(background_color)  # CopyI (copy entire input)
            selections.append([0, 0, output_h-1, output_w-1])  # Select entire input
            
            # Step 2: Copy entire input pattern once
            operations.append(28)  # CopyI (copy entire input)
            selections.append([0, 0, input_h-1, input_w-1])  # Select entire input
            
            # Step 3: Paste the pattern based on object positions in input
            for input_i in range(input_h):
                for input_j in range(input_w):
                    # Check if this position in input has object color
                    if input_grid[input_i, input_j] == object_color:
                        # Calculate the corresponding output position
                        paste_start_row = input_i * scale_factor
                        paste_start_col = input_j * scale_factor
                        paste_end_row = paste_start_row + input_h - 1
                        paste_end_col = paste_start_col + input_w - 1
                        
                        # Paste the entire input pattern at this position
                        operations.append(30)  # Paste
                        selections.append([paste_start_row, paste_start_col, paste_end_row, paste_end_col])
            
            # Step 4: Submit
            operations.append(34)  # Submit
            selections.append([0, 0, output_h-1, output_w-1])

            desc = {'id': f'007bbfb7-expert_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            
            # Create random trajectory that places patterns on background positions
            for i in range(1):
                selections_random, operations_random = self.random_trajectory(
                    input_h, input_w, output_h, output_w, background_color, object_color, pr_in[0])
                
                desc = {'id': f'007bbfb7-random_{num}_{i+1}',
                        'selections': selections_random.copy(),
                        'operations': operations_random.copy()}
                
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            
        return dat
    
    def random_trajectory(self, input_h, input_w, output_h, output_w, background_color, object_color, input_grid):
        selections = []
        operations = []
        scale_factor = input_h

        # Step 1: Resize grid
        operations.append(33)  # ResizeGrid
        selections.append([0, 0, output_h-1, output_w-1])

        # Step 2: Fill with background color
        operations.append(background_color)
        selections.append([0, 0, output_h-1, output_w-1])

        # Step 3: Copy entire input pattern
        operations.append(28)  # CopyI
        selections.append([0, 0, input_h-1, input_w-1])

        # Step 4: Mix of correct and incorrect positions
        # Correct positions: where object_color exists in input
        correct_positions = [(i, j) for i in range(input_h) for j in range(input_w)
                             if input_grid[i, j] == object_color]
        # Incorrect positions: where background_color exists in input
        incorrect_positions = [(i, j) for i in range(input_h) for j in range(input_w)
                               if input_grid[i, j] == background_color]

        # Randomly sample from correct positions (0 to all)
        num_correct = np.random.randint(0, len(correct_positions) + 1)
        sampled_correct = random.sample(correct_positions, num_correct) if num_correct > 0 else []

        # Randomly sample from incorrect positions (0 to all)
        num_incorrect = np.random.randint(0, len(incorrect_positions) + 1) if incorrect_positions else 0
        sampled_incorrect = random.sample(incorrect_positions, num_incorrect) if num_incorrect > 0 else []

        # Combine and shuffle
        selected_positions = sampled_correct + sampled_incorrect
        random.shuffle(selected_positions)

        for input_i, input_j in selected_positions:
            paste_start_row = input_i * scale_factor
            paste_start_col = input_j * scale_factor
            paste_end_row = paste_start_row + input_h - 1
            paste_end_col = paste_start_col + input_w - 1

            # Ensure paste area is within bounds
            if paste_end_row < output_h and paste_end_col < output_w:
                operations.append(30)  # Paste
                selections.append([paste_start_row, paste_start_col, paste_end_row, paste_end_col])

        # Step 5: Submit
        operations.append(34)  # Submit
        selections.append([0, 0, output_h-1, output_w-1])

        return selections, operations