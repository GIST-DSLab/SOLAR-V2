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
            p_color = random.choice(range(1, 10))

            # p_color를 제외한 나머지에서 4개 고르기
            color_list = list(range(1, 10))
            color_list.pop(color_list.index(p_color))
            four_colors = random.sample(color_list, 4)

            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                # 내 구현
                h = np.random.randint(10, max_h)
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 랜덤으로 n개 찍기
                num_p = random.choice(range(1, h * h // 16))

                points = []
                for _ in range(num_p):
                    x = random.randint(1, h-4)
                    y = random.randint(1, w-4)
                    new_point = (x, y)
                    if all(abs(new_point[0] - point[0]) >= 4 and abs(new_point[1] - point[1]) >= 4 for point in points):
                        points.append(new_point)
                    else:
                        continue

                for x, y in points:
                    rand_grid[x][y] = p_color
                    rand_grid[x+1][y] = p_color
                    rand_grid[x][y+1] = p_color
                    rand_grid[x+1][y+1] = p_color

                # answer - 3개에 대한 색이 같아야 한다. 문제 넣을 때도 주의하자!

                answer_grid = rand_grid.copy()

                # 색칠하기
                idx_list = [((-1, -1), four_colors[0]), ((-1, 2), four_colors[1]), ((2, -1), four_colors[2]), ((2, 2), four_colors[3])]
                random.shuffle(idx_list)

                for x, y in points:
                    for idx in idx_list:
                        answer_grid[x+idx[0][0]][y+idx[0][1]] = idx[1]
                        if (j == num_examples):
                            selections.append([x+idx[0][0], y+idx[0][1], 0, 0])
                            operations.append(idx[1])    # Color

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

            desc = {'id': '95990924',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
