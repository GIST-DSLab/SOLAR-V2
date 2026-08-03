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

            # 공통

            # 색상 3개 선택
            color_list = random.sample(range(1, 10), 2)

            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                # 내 구현
                h = np.random.randint(4, max_h)
                w = np.random.randint(4, max_w)
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 네모 범위 만들기
                p_x1 = np.random.randint(h)
                p_x2 = p_x1
                while abs(p_x1 - p_x2) < 3:
                    p_x1 = np.random.randint(h)
                    p_x2 = np.random.randint(h)
                if p_x1 > p_x2:
                    p_x1, p_x2 = p_x2, p_x1

                p_y1 = np.random.randint(w)
                p_y2 = p_y1
                while abs(p_y1 - p_y2) < 3:
                    p_y1 = np.random.randint(w)
                    p_y2 = np.random.randint(w)
                if p_y1 > p_y2:
                    p_y1, p_y2 = p_y2, p_y1

                # 색 선택
                p_color = color_list[0]
                rand_grid[p_x1, p_y1:p_y2+1] = p_color
                rand_grid[p_x2, p_y1:p_y2+1] = p_color
                rand_grid[p_x1:p_x2+1, p_y1] = p_color
                rand_grid[p_x1:p_x2+1, p_y2] = p_color

                # 한 개 구멍

                points = []
                for x in range(p_x1, p_x2+1):
                    for y in range(p_y1, p_y2+1):
                        if x == p_x1 or x == p_x2 or y == p_y1 or y == p_y2:
                            if (x == p_x1 and y == p_y1) or (x == p_x2 and y == p_y1) or (x == p_x1 and y == p_y2) or (x == p_x2 and y == p_y2):
                                continue
                            else:
                                points.append((x, y))

                hole = random.choice(points)
                h_x, h_y = hole
                rand_grid[h_x, h_y] = 0

                # answer + ARCLE - 빈 구멍으로 부터 색상 색칠 3개 동일

                answer_grid = rand_grid.copy()

                # 내부 색칠
                for x in range(p_x1+1, p_x2):
                    for y in range(p_y1+1, p_y2):
                        answer_grid[x][y] = color_list[1]

                if (j == num_examples):
                    selections.append([p_x1+1, p_y1+1, p_x2 - p_x1 - 2, p_y2 - p_y1 - 2])
                    operations.append(color_list[1])    # Color

                if h_y == p_y1:  # 왼쪽
                    answer_grid[h_x, :h_y+1] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, 0, 0, p_y1])
                        operations.append(color_list[1])    # Color
                elif h_y == p_y2:  # 오른쪽
                    answer_grid[h_x, h_y:] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, p_y2, 0, w - p_y2])
                        operations.append(color_list[1])    # Color
                elif h_x == p_x1:  # 위
                    answer_grid[:h_x+1, h_y] = color_list[1]
                    if (j == num_examples):
                        selections.append([0, h_y, p_x1, 0])
                        operations.append(color_list[1])    # Color
                elif h_x == p_x2:  # 아래
                    answer_grid[h_x:, h_y] = color_list[1]
                    if (j == num_examples):
                        selections.append([h_x, h_y, h - h_x, 0])
                        operations.append(color_list[1])    # Color

                # ARCLE

                if (j == num_examples):
                    operations.append(34)   # Submit
                    selections.append([0, 0, h-1, w-1])

                    pr_in.append(rand_grid)
                    pr_out.append(answer_grid)
                    j = j + 1

                # Example case 저장
                else:
                    ex_in.append(rand_grid)
                    ex_out.append(answer_grid)
                    j = j + 1

            desc = {'id': 'd4f3cd78',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
