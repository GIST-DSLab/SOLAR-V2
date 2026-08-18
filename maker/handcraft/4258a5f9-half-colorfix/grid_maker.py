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
            selections: List[NDArray] = []
            operations: List[NDArray] = []

            # # 색상 3개 선택
            # p_color = random.choice(range(1,10))
            # r_color = p_color
            # while r_color == p_color:
            #     r_color = np.random.choice(range(1,10))
            p_color = 5
            r_color = 1
 
            # 내 구현
            j = 0
            # num_examples 만큼 예제 만들기 + 마지막 1개 test case
            while (j < num_examples+1):            
                # 내 구현
                h = np.random.randint(5, max_h)
                w = h
                rand_grid = np.zeros((h, w), dtype=np.uint8)

                # 랜덤으로 n개 찍기
                num_p = random.choice(range(1, h * h // 9)) 

                points = []
                for _ in range(num_p):
                    x = random.randint(1,h-2)
                    y = random.randint(1,w-2)
                    new_point = (x, y)
                    if all(abs(new_point[0] - point[0]) >= 2 and abs(new_point[1] - point[1]) >= 2 for point in points):
                        points.append(new_point)
                    else:
                        continue

                for x, y in points:
                    rand_grid[x][y] = p_color

                # answer - 3개에 대한 색이 같아야 한다. 문제 넣을 때도 주의하자! 
                answer_grid = rand_grid.copy()

                for x, y in points:
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:                        answer_grid[x+dx][y+dy] = r_color

                # ARCLE 
                if (j == num_examples): 
                    choice = 0 
                    # choice = random.choice([0, 1, 2])  # 3가지 choice 중 랜덤 선택
                    
                    if choice == 0:
                        # 각 점 주위를 개별적으로 칠하기 (기존 방식)
                        all_cells = []
                        for x, y in points:
                            for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
                                all_cells.append((x+dx, y+dy))

                        # 랜덤 순서로 칠하기
                        # random.shuffle(all_cells)
                        for x, y in all_cells:
                            selections.append([x, y, 0, 0])           
                            operations.append(r_color)    # Color    
                    
                    # elif choice == 1:
                    #     # 3x3 영역으로 칠하기 (기존 방식)
                    #     point_order = points.copy()
                    #     random.shuffle(point_order)  # 점들을 랜덤 순서로 처리
                        
                    #     for x, y in point_order:
                    #         selections.append([x-1, y-1, 2, 2])           
                    #         operations.append(r_color)    # Color    
                    #         selections.append([x, y, 0, 0])           
                    #         operations.append(p_color)    # Color  
                    
                    # elif choice == 2:
                    #     # 각 변(상하좌우 + 대각선)을 따로 선택해서 칠하기
                    #     directions = [(1,1),(1,0),(1,-1),(0,1),(0,-1),(-1,1),(-1,0),(-1,-1)]
                        
                    #     for x, y in points:
                    #         # 각 방향에 대해 개별 셀 선택
                    #         direction_order = directions.copy()
                    #         random.shuffle(direction_order)  # 방향을 랜덤 순서로 처리
                            
                    #         for dx, dy in direction_order:
                    #             selections.append([x+dx, y+dy, 0, 0])
                    #             operations.append(r_color)
                            
                    #         # 중심점 다시 칠하기
                    #         selections.append([x, y, 0, 0])
                    #         operations.append(p_color)
                            
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

            desc = {'id': f'4258a5f9-gold_standard_{num}',
                    'selections': selections,
                    'operations': operations}
            dat.append((ex_in, ex_out, pr_in, pr_out, desc))
            
            # Random strategy: 중심점 주위 8칸이 아닌 다른 곳을 색칠
            for i in range(1):
                new_operations = []
                new_selections = []
                
                # 전체 그리드에서 색칠 가능한 위치들 수집 (중심점과 그 8칸 제외)
                forbidden_positions = set()
                for x, y in points:
                    forbidden_positions.add((x, y))  # 중심점
                    for dx, dy in [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]:
                        if 0 <= x+dx < h and 0 <= y+dy < w:
                            forbidden_positions.add((x+dx, y+dy))  # 주위 8칸
                
                # 색칠 가능한 위치들
                available_positions = []
                for x in range(h):
                    for y in range(w):
                        if (x, y) not in forbidden_positions:
                            available_positions.append((x, y))
                
                if available_positions:
                    # Random coloring strategies - 모든 전략에서 r_color 고정 사용
                    strategy = 0
                    
                    if strategy == 0:
                        # Strategy 1: 랜덤한 위치들에 r_color로 색칠
                        num_colors = min(np.random.randint(5, 15), len(available_positions))
                        selected_positions = random.sample(available_positions, num_colors)
                        
                        for x, y in selected_positions:
                            rand_color=np.random.choice([0,r_color])
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(rand_color)
                    
                    elif strategy == 1:
                        # Strategy 2: 코너나 모서리 위주로 r_color로 색칠
                        corner_edge_positions = []
                        for x, y in available_positions:
                            # 코너나 모서리 위치 확인
                            if (x == 0 or x == h-1 or y == 0 or y == w-1):
                                corner_edge_positions.append((x, y))
                        
                        if corner_edge_positions:
                            num_colors = min(np.random.randint(3, 8), len(corner_edge_positions))
                            selected_positions = random.sample(corner_edge_positions, num_colors)
                        else:
                            num_colors = min(np.random.randint(3, 8), len(available_positions))
                            selected_positions = random.sample(available_positions, num_colors)
                        
                        for x, y in selected_positions:
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(r_color)
                    
                    else:
                        # Strategy 3: 중심점들과 거리가 먼 곳들을 r_color로 색칠
                        distant_positions = []
                        for x, y in available_positions:
                            # 가장 가까운 중심점과의 거리 계산
                            min_distance = float('inf')
                            for px, py in points:
                                distance = abs(x - px) + abs(y - py)  # 맨하탄 거리
                                min_distance = min(min_distance, distance)
                            
                            # 거리 3 이상인 위치들만 선택
                            if min_distance >= 3:
                                distant_positions.append((x, y))
                        
                        if distant_positions:
                            num_colors = min(np.random.randint(3, 10), len(distant_positions))
                            selected_positions = random.sample(distant_positions, num_colors)
                        else:
                            # distant_positions가 없으면 일반 available_positions 사용
                            num_colors = min(np.random.randint(3, 8), len(available_positions))
                            selected_positions = random.sample(available_positions, num_colors)
                        
                        for x, y in selected_positions:
                            new_selections.append([x, y, 0, 0])
                            new_operations.append(r_color)

                # Submit 추가
                new_operations.append(34)
                new_selections.append([0, 0, h-1, w-1])

                desc = {
                    'id': f'4258a5f9-random_{num}_{i}',
                    'selections': new_selections,
                    'operations': new_operations,
                }
                dat.append((ex_in, ex_out, pr_in, pr_out, desc))
        return dat