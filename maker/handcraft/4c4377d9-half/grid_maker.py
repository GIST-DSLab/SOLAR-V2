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
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            operations = []
            selections = []

            h = np.random.randint(3, max_h//2+1)
            w = h

            answer_h = 2 * h
            answer_w = w

            numbers = list(range(1,10))

            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            for _ in range(num_examples):
                selected_numbers = random.sample(numbers, 2)
                rand_grid = np.random.choice(selected_numbers, size=[h, w])
                # answer
                down_grid = np.flipud(rand_grid)
                answer_grid = np.concatenate((rand_grid, down_grid))
                ex_in.append(down_grid)
                ex_out.append(answer_grid)

            selected_numbers = random.sample(numbers, 2)
            rand_grid = np.random.choice(selected_numbers, size=[h, w])
            # answer
            down_grid = np.flipud(rand_grid)
            answer_grid = np.concatenate((rand_grid, down_grid))
            pr_in.append(down_grid)
            pr_out.append(answer_grid)

            selections_expert, operations_expert = self.expert_trajectory(
                h, w, answer_h, answer_w)

            desc = {'id': f'4c4377d9-expert_{num}',
                    'selections': selections_expert.copy(),
                    'operations': operations_expert.copy()}

            dat.append((ex_in, ex_out, pr_in, pr_out, desc))

            for i in range(1):
                selections_random, operations_random = self.random_trajectory(
                    h, w, answer_h, answer_w, num, i, num_operations=8)

                desc = {'id': f'4c4377d9-random_{num}_{i+1}',
                        'selections': selections_random.copy(),
                        'operations': operations_random.copy()}

                dat.append((ex_in, ex_out, pr_in, pr_out, desc))

        return dat

    def expert_trajectory(self, h, w, answer_h, answer_w):
        selections = []
        operations = []

        selections.append([0, 0, answer_h-1, answer_w-1])
        operations.append(33)   # Crop Grid

        selections.append([0, 0, h-1, w-1])
        operations.append(29)   # CopyO

        selections.append([h, 0, h-1, w-1])
        operations.append(30)   # Paste

        selections.append([0, 0, h-1, w-1])
        operations.append(27)   # FlipV

        selections.append([0, 0, answer_h-1, answer_w-1])
        operations.append(34)  # Submit

        return selections, operations

    def random_trajectory(self, h, w, answer_h, answer_w, num, i,  num_operations=3):
        selections = []
        operations = []

        selections.append([0, 0, answer_h-1, answer_w-1])
        operations.append(33)   # Crop Grid

        selections.append([0, 0, h-1, w-1])
        operations.append(29)   # CopyO

        selections.append([h, 0, h-1, w-1])
        operations.append(30)   # Paste

        selections.append([0, 0, h-1, w-1])
        operations.append(27)   # FlipV

        state = np.random.get_state()
        np.random.seed(num+i)
        insertion_point = np.random.choice([0,2,3])
        # print(insertion_point,end=" ")
        selections[:] = selections[:insertion_point + 1]
        operations[:] = operations[:insertion_point + 1]

        # Possible random operations to choose from
        if insertion_point == 0:
            random_ops = [24, 25, 26, 27, 29]
            below = False
            
            for _ in range(num_operations):
            # Perform a random operation
                random_op = random.choice(random_ops)
                if random_op == 29 :
                    if below:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                            [h, 0, h-1, w-1]
                        ]
                    else:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                        ]
                    below = True
                elif random_ops == 26 or random_ops ==27:
                    if below:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                            [h, 0, h-1, w-1],
                            [0, 0, answer_h-1, answer_w-1]
                        ]
                    else:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                            [0, 0, answer_h-1, answer_w-1]
                        ]
                else:
                    if below:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                            [h, 0, h-1, w-1]
                        ]
                    else:
                        possible_selections = [
                            [0, 0, h-1, w-1],
                        ]
                
                selection = random.choice(possible_selections)

                operations.append(random_op)
                selections.append(selection)

            if random_op == 29:
                if selection == [0, 0, h-1, w-1]:
                    paste_selection = [h, 0, h-1, w-1]
                else: 
                    paste_selection = [0, 0, h-1, w-1]
                operations.append(30)
                selections.append(paste_selection)
            
        else :
            random_ops = [24, 25, 26, 27]
            for _ in range(num_operations):
            # Perform a random operation
                random_op = random.choice(random_ops)
                if random_ops == 26 or random_ops ==27:
                    possible_selections = [
                        [0, 0, h-1, w-1],
                        [h, 0, h-1, w-1],
                        [0, 0, answer_h-1, answer_w-1]
                    ]
                else:
                    possible_selections = [
                        [0, 0, h-1, w-1],
                        [h, 0, h-1, w-1]
                    ]
                    
                selection = random.choice(possible_selections)

                operations.append(random_op)
                selections.append(selection)

        selections.append([0, 0, answer_h-1, answer_w-1])
        operations.append(34)  # Submit

        np.random.set_state(state)
        
        return selections, operations