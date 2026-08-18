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

        # 색상 매핑 정의 (1-9 범위의 색상만 사용)
        available_colors = list(range(1, 10))
        
        # randomly generate inputs
        while num < num_samples:
            num += 1
            pr_in: List[NDArray] = []
            pr_out: List[NDArray] = []
            ex_in: List[NDArray] = []
            ex_out: List[NDArray] = []

            operations = []
            selections = []

            # example case들에서 관찰된 색상 변환 규칙 저장
            observed_color_mapping = {}
            
            # Example cases 생성
            for j in range(num_examples):
                # 그리드 크기 결정 (2x2 ~ max_h x max_w)
                grid_size = np.random.randint(2, max_h + 1)
                h = grid_size
                w = grid_size
                
                # 입력 그리드 생성
                input_grid = np.zeros((h, w), dtype=np.uint8)
                
                # 각 열(column)을 하나의 색상으로 칠하기
                input_colors_in_example = []
                for col in range(w):
                    # 사용할 색상 선택 (1~9)
                    color = random.choice(available_colors)
                    input_grid[:, col] = color
                    if color not in input_colors_in_example:
                        input_colors_in_example.append(color)
                
                # 이 example에서 사용된 색상들에 대한 변환 규칙 생성
                available_target_colors = available_colors.copy()
                # 이미 사용된 target 색상들 제거
                used_target_colors = list(observed_color_mapping.values())
                for used_color in used_target_colors:
                    if used_color in available_target_colors:
                        available_target_colors.remove(used_color)
                
                # 새로운 색상 매핑 생성
                for input_color in input_colors_in_example:
                    if input_color not in observed_color_mapping:
                        # 사용 가능한 색상이 있으면 선택, 없으면 랜덤 선택
                        if available_target_colors:
                            target_color = random.choice(available_target_colors)
                            available_target_colors.remove(target_color)
                        else:
                            # 사용 가능한 색상이 없으면 전체 색상에서 랜덤 선택
                            target_color = random.choice(available_colors)
                        observed_color_mapping[input_color] = target_color
                
                # 출력 그리드 생성
                output_grid = np.zeros((h, w), dtype=np.uint8)
                for col in range(w):
                    original_color = input_grid[0, col]  # 해당 column의 색상
                    new_color = observed_color_mapping[original_color]
                    output_grid[:, col] = new_color  # 전체 column을 새 색상으로 칠하기

                ex_in.append(input_grid)
                ex_out.append(output_grid)
            
            # Test case 생성
            if observed_color_mapping:  # 매핑이 있는 경우에만
                # 그리드 크기 결정
                grid_size = np.random.randint(2, max_h + 1)
                h = grid_size
                w = grid_size
                
                # test case 그리드 생성 (관찰된 색상만 사용)
                input_grid = np.zeros((h, w), dtype=np.uint8)
                observed_input_colors = list(observed_color_mapping.keys())
                
                for col in range(w):
                    # example에서 나온 입력 색상만 사용
                    color = random.choice(observed_input_colors)
                    input_grid[:, col] = color
                
                # 출력 그리드 생성 (관찰된 변환 규칙만 적용)
                output_grid = np.zeros((h, w), dtype=np.uint8)
                for col in range(w):
                    original_color = input_grid[0, col]  # 해당 column의 색상
                    new_color = observed_color_mapping[original_color]
                    output_grid[:, col] = new_color  # 전체 column을 새 색상으로 칠하기
                    
                    # 각 column에 대한 선택과 색상 변경 작업
                    selections.append([0, col, h-1, 0])  # 전체 column 선택
                    operations.append(new_color)

                operations.append(34)   # Submit
                selections.append([0, 0, h-1, w-1])

                pr_in.append(input_grid)
                pr_out.append(output_grid)

            desc = {'id': f'0d3d703e-expert_{num}',
                    'selections': selections,
                    'operations': operations,
                    'observed_color_mapping': observed_color_mapping}

            dat.append((ex_in, ex_out, pr_in, pr_out, desc))

            # random trajectory 추가 (매핑과 다른 색상 선택)
            for i in range(1):
                selections_random, operations_random = self.random_trajectory(
                    h, w, input_grid, observed_color_mapping, available_colors)

                desc = {'id': f'0d3d703e-random_{num}_{i+1}',
                        'selections': selections_random.copy(),
                        'operations': operations_random.copy(),
                        'observed_color_mapping': observed_color_mapping}

                dat.append((ex_in, ex_out, pr_in, pr_out, desc))

        return dat

    def random_trajectory(self, h, w, input_grid, observed_color_mapping, available_colors):
        """정답을 따르다가 중간에 branch하여 틀린 색상을 선택하는 random trajectory 생성
        - 순서는 정답과 동일 (왼쪽 column부터)
        - 랜덤한 지점까지는 정답대로, 그 이후부터는 틀린 색상 선택
        """
        selections = []
        operations = []

        # branch 지점 선택 (0 ~ w-1, 최소 1개는 틀리도록)
        branch_point = random.randint(0, w - 1)

        for col in range(w):
            original_color = input_grid[0, col]
            correct_color = observed_color_mapping[original_color]

            if col < branch_point:
                # branch 전: 정답대로
                selections.append([0, col, h-1, 0])
                operations.append(correct_color)
            else:
                # branch 후: 완전 랜덤 색상 선택 (정답일 수도 있음)
                random_color = random.choice(available_colors)

                selections.append([0, col, h-1, 0])
                operations.append(random_color)

        # Submit
        operations.append(34)
        selections.append([0, 0, h-1, w-1])

        return selections, operations