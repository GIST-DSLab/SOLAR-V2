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

            # 공통
            p_color = np.random.randint(1, 10)       # center 색상
            w_color = np.random.randint(1, 10)
            while w_color == p_color:           # 색상 달라야함
                w_color = np.random.randint(1, 10)

            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):
                # 내 구현
                h = np.random.randint(7, max_h)
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)        # 모두 검은 배경

                # 왼쪽위에 치우친 center 만들기
                p_x = np.random.randint(2, h // 2)
                p_y = np.random.randint(2, h // 2)
                rand_grid[p_x][p_y] = p_color

                # 날개 만들기
                wing_grid = np.random.randint(0, 2, size=[p_x, p_y], dtype=np.uint8) * w_color

                # 적어도 두개는 색이 있도록 함
                for _ in range(2):
                    x, y = np.random.randint(0, p_x), np.random.randint(0, p_y)
                    wing_grid[x, y] = w_color

                random_int = np.random.randint(4)   # 상하좌우 선택

                # 정답
                wing_ud = np.flipud(wing_grid)
                wong_lr = np.fliplr(wing_grid)
                right_ud_lr = np.fliplr(wing_ud)

                # 나중에 if 넣어주기
                answer_grid = rand_grid.copy()
                choice = np.random.randint(4)

                # 좌상
                if choice == 0:
                    for x in range(len(wing_grid)):
                        for y in range(len(wing_grid[0])):
                            rand_grid[x][y] = wing_grid[x][y]  # 좌상

                            answer_grid[x][y] = wing_grid[x][y]  # 좌상
                            answer_grid[p_x+x+1][y] = wing_ud[x][y]
                            answer_grid[x][p_y+1+y] = wong_lr[x][y]
                            answer_grid[p_x+x+1][p_y+1+y] = right_ud_lr[x][y]
                    if (j == num_examples):
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(28)    # CopyInput
                        selections.append([p_x + 1, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x + 1, 0, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV
                        selections.append([0, p_y + 1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, p_y + 1, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([p_x + 1, p_y + 1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x + 1, p_y + 1, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([p_x + 1, p_y + 1, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV

                elif choice == 1:
                    # 우상
                    for x in range(len(wing_grid)):
                        for y in range(len(wing_grid[0])):
                            rand_grid[x][y+p_y+1] = wing_grid[x][y]  # 좌상

                            answer_grid[x][y+p_y+1] = wing_grid[x][y]  # 좌상
                            answer_grid[p_x+x+1][p_y+y+1] = wing_ud[x][y]
                            answer_grid[x][y] = wong_lr[x][y]
                            answer_grid[p_x+x+1][y] = right_ud_lr[x][y]
                    if (j == num_examples):
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(28)    # CopyInput
                        selections.append([p_x + 1, p_y+1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x + 1, p_y+1, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([p_x + 1, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x + 1, 0, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([p_x + 1, 0, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV

                elif choice == 2:
                    # 좌하
                    for x in range(len(wing_grid)):
                        for y in range(len(wing_grid[0])):
                            rand_grid[x+p_x+1][y] = wing_grid[x][y]  # 좌상

                            answer_grid[x+p_x+1][y] = wing_grid[x][y]  # 좌상
                            answer_grid[x][y] = wing_ud[x][y]
                            answer_grid[p_x+x+1][p_y+y+1] = wong_lr[x][y]
                            answer_grid[x][p_y+y+1] = right_ud_lr[x][y]
                    if (j == num_examples):
                        selections.append([p_x+1, 0, p_x - 1, p_y - 1])
                        operations.append(28)    # CopyInput
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV
                        selections.append([p_x+1, p_y+1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x+1, p_y+1, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV

                elif choice == 3:
                    # 우하
                    for x in range(len(wing_grid)):
                        for y in range(len(wing_grid[0])):
                            rand_grid[x+p_x+1][y+p_y+1] = wing_grid[x][y]  # 좌상

                            answer_grid[x+p_x+1][y+p_y+1] = wing_grid[x][y]  # 좌상
                            answer_grid[x][p_y+y+1] = wing_ud[x][y]
                            answer_grid[x+p_x+1][y] = wong_lr[x][y]
                            answer_grid[x][y] = right_ud_lr[x][y]
                    if (j == num_examples):
                        selections.append([p_x+1, p_y+1, p_x - 1, p_y - 1])
                        operations.append(28)    # CopyInput
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, p_y+1, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV
                        selections.append([p_x+1, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([p_x+1, 0, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(30)    # Paste
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(26)    # FlipH
                        selections.append([0, 0, p_x - 1, p_y - 1])
                        operations.append(27)    # FlipV

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

            desc = {'id': '4938f0c2',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat
