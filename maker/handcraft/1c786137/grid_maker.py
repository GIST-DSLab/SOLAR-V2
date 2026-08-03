from maker.base_grid_maker import BaseGridMaker
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np


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

            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                h = np.random.randint(6, max_h)
                w = np.random.randint(6, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 1부터 9까지의 숫자 리스트 생성
                numbers = list(range(10))

                # 무작위로 세 숫자 선택
                selected_numbers = np.random.choice(numbers, 4, replace=False)

                # 3가지 숫자를 랜덤하게 선택하여 배열에 채우기
                for x in range(h):
                    for y in range(w):
                        rand_grid[x, y] = np.random.choice(selected_numbers)

                # 선 그을 곳 찾기
                l_y_1 = np.random.randint(1, w-1)
                l_y_2 = np.random.randint(1, w-1)
                l_x_1 = np.random.randint(1, h-1)
                l_x_2 = np.random.randint(1, h-1)
                while abs(l_y_2 - l_y_1) < 2:
                    l_y_2 = np.random.randint(1, w-1)
                while abs(l_x_2 - l_x_1) < 2:
                    l_x_2 = np.random.randint(1, h-1)

                if l_x_1 > l_x_2:
                    l_x_2, l_x_1 = l_x_1, l_x_2
                if l_y_1 > l_y_2:
                    l_y_2, l_y_1 = l_y_1, l_y_2

                # 선 색상
                unselected_numbers = [x for x in numbers if x not in selected_numbers]
                l_color = np.random.choice(unselected_numbers)

                # 색칠c:\Users\HeeJunKim\Downloads\generate_trajectory.py
                for x in range(len(rand_grid)):
                    for y in range(len(rand_grid[0])):
                        # 위에 bar
                        if x == l_x_1 and y >= l_y_1 and y <= l_y_2:
                            rand_grid[x, y] = l_color

                        if x == l_x_2 and y >= l_y_1 and y <= l_y_2:
                            rand_grid[x, y] = l_color

                        if y == l_y_1 and x >= l_x_1 and x <= l_x_2:
                            rand_grid[x, y] = l_color

                        if y == l_y_2 and x >= l_x_1 and x <= l_x_2:
                            rand_grid[x, y] = l_color

                # answer
                answer_grid = rand_grid.copy()
                answer_grid = rand_grid[l_x_1+1: l_x_2, l_y_1+1: l_y_2]

                # ARCLE
                if (j == num_examples):

                    # Output Grid 지정
                    answer_h = l_x_2 - l_x_1 - 1
                    answer_w = l_y_2 - l_y_1 - 1

                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(33)    # CropGrid

                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(32)    # Reset Grid

                    # InputCopy
                    selections.append([l_x_1+1, l_y_1+1, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(28)    # CopyInput

                    # OutputPaste
                    selections.append([0, 0, l_x_2 - l_x_1 - 2, l_y_2 - l_y_1 - 2])
                    operations.append(30)    # Paste

                    operations.append(34)   # Submit
                    selections.append([0, 0, answer_h-1, answer_w-1])

                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1

                # Example case 저장
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1

            desc = {'id': '1c786137',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
