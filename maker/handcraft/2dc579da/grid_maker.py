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
                h = np.random.randint(5, max_h)
                w = np.random.randint(5, max_w)

                # 배경색
                b_color = np.random.randint(1, 10)
                rand_grid = np.full((h, w), b_color, dtype=np.uint8)

                # 선 그을 곳 찾기
                l_y = np.random.randint(1, w-1)
                l_x = np.random.randint(1, h-1)
                l_color = b_color
                while l_color == b_color:
                    l_color = np.random.randint(1, 10)

                # 선 긋기
                rand_grid[l_x] = l_color
                rand_grid[:, l_y] = l_color

                # 점찍기
                p_color = np.random.randint(1, 10)
                while p_color == b_color or p_color == l_color:
                    p_color = np.random.randint(1, 10)

                p_y = np.random.randint(w)
                while p_y == l_y or p_y == l_x:
                    p_y = np.random.randint(w)

                p_x = np.random.randint(h)
                while p_x == l_y or p_x == l_x:
                    p_x = np.random.randint(h)

                rand_grid[p_x, p_y] = p_color

                # answer - 점 찾기 (개수로 찾기) + ARCLE
                answer_grid = rand_grid.copy()

                color_dict = {key: 0 for key in range(1, 10)}
                for x in range(len(rand_grid)):
                    for y in range(len(rand_grid[0])):
                        value = rand_grid[x][y]
                        color_dict[value] += 1

                non_zero_color_dict = {key: value for key, value in color_dict.items() if value != 0}
                min_color = min(non_zero_color_dict, key=non_zero_color_dict.get)

                for x in range(len(rand_grid)):
                    for y in range(len(rand_grid[0])):
                        if rand_grid[x][y] == min_color:
                            point_x = x
                            point_y = y

                # 좌상
                if point_x < l_x and point_y < l_y:
                    answer_grid = rand_grid[:l_x, :l_y]
                    if (j == num_examples):
                        answer_h, answer_w = l_x, l_y

                        selections.append([0, 0, answer_h - 1, answer_w - 1])
                        operations.append(33)    # CropGrid

                        selections.append([0, 0, answer_h - 1, answer_w - 1])
                        operations.append(28)    # CopyInput

                # 우상
                elif point_x < l_x and point_y > l_y:
                    answer_grid = rand_grid[:l_x, l_y+1:]
                    if (j == num_examples):
                        answer_h, answer_w = l_x, w - l_y - 1

                        selections.append([0, 0, answer_h - 1, answer_w - 1])
                        operations.append(33)    # CropGrid

                        selections.append([0, l_y+1, answer_h - 1, answer_w - 1])
                        operations.append(28)    # CopyInput

                # 좌하
                elif point_x > l_x and point_y < l_y:
                    answer_grid = rand_grid[l_x+1:, :l_y]
                    if (j == num_examples):
                        answer_h, answer_w = h - l_x - 1, l_y

                        selections.append([0, 0, answer_h - 1, answer_w - 1])
                        operations.append(33)    # CropGrid

                        selections.append([l_x+1, 0, answer_h - 1, answer_w - 1])
                        operations.append(28)    # CopyInput

                # 우하
                elif point_x > l_x and point_y > l_y:
                    answer_grid = rand_grid[l_x+1:, l_y+1:]
                    if (j == num_examples):
                        answer_h, answer_w = h - l_x - 1, w - l_y - 1

                        selections.append([0, 0, answer_h - 1, answer_w - 1])
                        operations.append(33)    # CropGrid

                        selections.append([l_x+1, l_y+1, answer_h - 1, answer_w - 1])
                        operations.append(28)    # CopyInput

                # ARCLE
                if (j == num_examples):

                    # OutputPaste
                    selections.append([0, 0, answer_h - 1, answer_w - 1])
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

            desc = {'id': '2dc579da',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
