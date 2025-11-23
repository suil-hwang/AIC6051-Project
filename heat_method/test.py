import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import potpourri3d as pp3d
import torch
from neural_astar.utils.data import create_warcraft_dataloader
from scipy.ndimage import distance_transform_edt


def to_numpy(array):
    if torch.is_tensor(array):
        return array.detach().cpu().numpy()
    return np.asarray(array)

# --- 1. Mesh 변환 함수 (Numpy Array 입력용으로 수정) ---
def array_to_mesh(binary_map_2d):
    """
    2D 이진 맵(1: 이동가능, 0: 장애물)을 입력받아
    potpourri3d에서 사용할 수 있는 Mesh(Vertices, Faces)로 변환
    """
    rows, cols = binary_map_2d.shape
    
    # Vertex ID 매핑 (각 픽셀의 고유 ID, 장애물은 -1)
    vertex_ids = np.full((rows, cols), -1, dtype=int)
    vertices = []
    curr_id = 0
    
    # 1. 유효한 픽셀(이동 가능 영역)을 정점으로 생성
    for r in range(rows):
        for c in range(cols):
            if binary_map_2d[r, c] == 1: # 1 = Passable
                vertex_ids[r, c] = curr_id
                # (row, col) -> (x, y) 좌표계로 변환 (row=y, col=x)
                # 여기서는 시각화 편의를 위해 (r, c, 0) 그대로 사용
                vertices.append([r, c, 0]) 
                curr_id += 1
                
    vertices = np.array(vertices, dtype=np.float64)
    
    # 2. 인접한 정점들을 연결하여 Face(삼각형) 생성
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            # 2x2 격자 확인
            # v_tl -- v_tr
            #  |      |
            # v_bl -- v_br
            v_tl = vertex_ids[r, c]
            v_tr = vertex_ids[r, c+1]
            v_bl = vertex_ids[r+1, c]
            v_br = vertex_ids[r+1, c+1]
            
            # 상단 삼각형 (Top-Left, Bottom-Left, Top-Right)
            if v_tl != -1 and v_bl != -1 and v_tr != -1:
                faces.append([v_tl, v_bl, v_tr])
            
            # 하단 삼각형 (Top-Right, Bottom-Left, Bottom-Right)
            if v_tr != -1 and v_bl != -1 and v_br != -1:
                faces.append([v_tr, v_bl, v_br])
                
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces, vertex_ids

# --- 2. 고립된 Vertex 제거 함수 ---
def remove_unreferenced_vertices(vertices, faces, vertex_ids):
    """
    메쉬에서 어떤 face에도 참조되지 않는 정점을 제거하고,
    face 인덱스를 재조정하여 올바른 메쉬를 반환
    """
    # 어떤 face에도 사용되는 vertex ID 찾기
    referenced_vertices = np.unique(faces.flatten())
    
    # 새로운 vertex 배열 생성 (referenced만)
    new_vertices = vertices[referenced_vertices]
    
    # 새로운 vertex_ids 매핑 (old_id -> new_id)
    old_to_new = np.full(len(vertices), -1, dtype=int)
    old_to_new[referenced_vertices] = np.arange(len(referenced_vertices))
    
    # Face 인덱스 재조정
    new_faces = old_to_new[faces]
    
    # vertex_ids 2D 배열도 재조정
    new_vertex_ids = np.where(vertex_ids != -1, old_to_new[vertex_ids], -1)
    
    return new_vertices, new_faces, new_vertex_ids

# --- 3. 가장 가까운 유효 정점 찾기 ---
def get_nearest_valid_vertex(target_r, target_c, vertex_ids):
    """목표 좌표가 장애물일 경우 가장 가까운 빈 공간의 Vertex ID 반환"""
    rows, cols = vertex_ids.shape
    r = np.clip(target_r, 0, rows-1)
    c = np.clip(target_c, 0, cols-1)
    
    if vertex_ids[r, c] != -1:
        return vertex_ids[r, c]
    
    # 주변 탐색 (BFS 방식이 정확하나 여기선 간단히 거리순 탐색)
    indices = np.argwhere(vertex_ids != -1)
    if len(indices) == 0: return -1
    
    dists = (indices[:, 0] - r)**2 + (indices[:, 1] - c)**2
    nearest_idx = np.argmin(dists)
    r_near, c_near = indices[nearest_idx]
    
    return vertex_ids[r_near, c_near]

# --- 4. 출발점 생성 함수 (MazeDataset 로직 차용) ---
def get_random_start_node(opt_dist_map):
    """
    MazeDataset의 로직을 참고하여 목표 지점으로부터 일정 거리 떨어진 출발점을 랜덤하게 선택
    """
    od_vct = opt_dist_map.flatten()
    # 유효한 거리 값만 추출 (최소값보다 큰 값들)
    od_vals = od_vct[od_vct > od_vct.min()]
    
    if len(od_vals) == 0:
        return None
        
    # 난이도 구간 설정 (MazeDataset 기본값: 0.55, 0.70, 0.85)
    pcts = np.array([0.55, 0.70, 0.85, 1.0])
    
    # 상위 % 거리값들을 임계값으로 계산
    od_th = np.percentile(od_vals, 100.0 * (1 - pcts))
    
    # 구간 랜덤 선택
    r = np.random.randint(0, len(od_th) - 1)
    
    # 해당 구간에 속하는 인덱스들 찾기
    start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
    candidate_indices = np.where(start_candidate)[0]
    
    if len(candidate_indices) == 0:
        return None
        
    start_idx = np.random.choice(candidate_indices)
    
    # 1D index -> 2D coords
    rows, cols = opt_dist_map.shape
    start_r, start_c = divmod(start_idx, cols)
    
    return start_r, start_c

# --- 5. 데이터 로드 및 필드 계산 메인 함수 ---
def main():
    parser = argparse.ArgumentParser(description="Heat Method & Obstacle Distance Field Test")
    parser.add_argument("--mode", choices=["maze", "warcraft"], default="warcraft", help="Dataset type to visualize")
    parser.add_argument("--sample_idx", type=int, default=0, help="Index of the sample to visualize")
    parser.add_argument("--show_start", action="store_true", help="Generate and show a random start point")
    parser.add_argument("--data_path", type=str, default="data/maze/mixed_064_moore_c16.npz", help="Path to the .npz dataset (maze mode)")
    parser.add_argument("--warcraft_dir", type=str, default="data/warcraft_shortest_path/18x18", help="Directory with WarCraft *_maps.npy files")
    parser.add_argument("--warcraft_split", type=str, default="test", help="WarCraft split prefix (e.g., train, val, test)")
    args = parser.parse_args()

    if args.mode == "warcraft":
        try:
            dataloader = create_warcraft_dataloader(
                args.warcraft_dir,
                args.warcraft_split,
                batch_size=1,
                shuffle=False,
            )
        except Exception as e:
            print(f"Error loading WarCraft data: {e}")
            return

        dataset = dataloader.dataset
        if args.sample_idx >= len(dataset):
            print(f"Error: Sample index {args.sample_idx} is out of bounds (0-{len(dataset)-1}).")
            return

        input_tensor, start_map, goal_map, opt_traj = dataset[args.sample_idx]
        input_np = to_numpy(input_tensor)
        start_map_np = to_numpy(start_map)
        goal_map_np = to_numpy(goal_map)

        rgb_image = np.clip(np.transpose(input_np[:3], (1, 2, 0)), 0, 1)
        geo_dist_map = np.squeeze(input_np[3])
        obs_dist_map = np.squeeze(input_np[4])

        g_coords = np.argwhere(goal_map_np.squeeze() == 1)
        start_coords = np.argwhere(start_map_np.squeeze() == 1)
        start_node = tuple(start_coords[0]) if args.show_start and len(start_coords) > 0 else None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb_image)
        axes[0].set_title("WarCraft RGB Map")
        if len(g_coords) > 0:
            g_r, g_c = g_coords[0]
            axes[0].scatter(g_c, g_r, c="red", s=80, label="Goal", edgecolors="white")
        if start_node:
            s_r, s_c = start_node
            axes[0].scatter(s_c, s_r, c="green", s=80, label="Start", edgecolors="white")
        axes[0].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend()

        cmap_geo = plt.cm.jet
        cmap_geo.set_bad(color="black")
        im1 = axes[1].imshow(geo_dist_map, cmap=cmap_geo)
        axes[1].set_title("Geodesic Distance Field\n(Heat Method)")
        plt.colorbar(im1, ax=axes[1])
        axes[1].axis("off")

        im2 = axes[2].imshow(obs_dist_map, cmap="viridis")
        axes[2].set_title("Obstacle Distance Field\n(Clearance)")
        plt.colorbar(im2, ax=axes[2])
        axes[2].axis("off")

        fig.tight_layout()
        plt.show()

        print("Done.")
        return

    # 데이터셋 로드
    data_path = args.data_path
    try:
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            # Neural A* 데이터셋 구조:
            # arr_0: maps (N, 1, H, W) -> 1:passable, 0:obstacle
            # arr_1: goals (N, 1, H, W)
            # arr_3: opt_dists (N, 1, H, W) -> Goal까지의 최적 거리 (출발점 생성용)
            maps = data['arr_0']
            goals = data['arr_1']
            if 'arr_3' in data:
                opt_dists = data['arr_3']
            else:
                opt_dists = None
        elif data_path.endswith('.npy'):
            # Warcraft 데이터셋 처리
            if "warcraft" in data_path.lower():
                dirname = os.path.dirname(data_path)
                basename = os.path.basename(data_path)
                prefix = basename.split('_')[0] # 'test', 'train', 'val'
                
                # Grid Map (Weights) 로드
                weights_path = os.path.join(dirname, f"{prefix}_vertex_weights.npy")
                if os.path.exists(weights_path):
                    print(f"Detected Warcraft dataset. Loading grid from: {weights_path}")
                    weights = np.load(weights_path) # (N, 12, 12)
                    # Warcraft는 모든 타일이 이동 가능(비용 차이)하므로 1로 설정
                    maps = np.ones_like(weights, dtype=np.float32)
                    
                    # Goal 로드 시도 (Shortest Paths)
                    paths_path = os.path.join(dirname, f"{prefix}_shortest_paths.npy")
                    if os.path.exists(paths_path):
                        paths = np.load(paths_path) # (N, 12, 12)
                        goals = np.zeros_like(paths)
                        # 각 경로의 마지막 점을 목표로 가정 (또는 임의의 점)
                        # 여기서는 간단히 경로 상의 점 중 하나를 목표로 설정하거나, 
                        # 추후 로직에서 목표가 없으면 랜덤 생성하도록 함.
                        # 일단 paths를 goals로 사용하지 않음 (paths는 전체 경로이므로)
                    else:
                        goals = np.zeros_like(maps)
                    
                    opt_dists = None
                else:
                    print(f"Warning: Warcraft weights file not found. Loading {data_path} directly.")
                    maps = np.load(data_path)
                    goals = np.zeros_like(maps)
                    opt_dists = None
            else:
                # 일반 .npy 파일
                maps = np.load(data_path)
                goals = np.zeros_like(maps)
                opt_dists = None
        else:
            print(f"Error: Unsupported file format '{data_path}'")
            return

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 샘플 선택
    sample_idx = args.sample_idx
    if sample_idx >= len(maps):
        print(f"Error: Sample index {sample_idx} is out of bounds (0-{len(maps)-1}).")
        return
    
    # 차원 정리: (1, 32, 32) -> (32, 32)
    map_design = maps[sample_idx].squeeze() 
    goal_map = goals[sample_idx].squeeze()
    
    rows, cols = map_design.shape
    print(f"Processing Sample {sample_idx}: Size {rows}x{cols}")

    # ==========================================
    # A. Geodesic Distance Field (Heat Method)
    # ==========================================
    print("Computing Geodesic Distance Field (Heat Method)...")
    
    # 1. Mesh 변환
    V, F, v_ids = array_to_mesh(map_design)
    
    if len(F) == 0:
        print("Error: 유효한 메쉬를 생성할 수 없습니다 (빈 맵?).")
        return

    # 1-1. 고립된 정점 제거 (potpourri3d 호환성)
    V, F, v_ids = remove_unreferenced_vertices(V, F, v_ids)
    print(f"Mesh: {len(V)} vertices, {len(F)} faces")

    # 2. 목표 지점의 Vertex ID 찾기
    # goal_map에서 값이 1인 좌표 찾기
    g_coords = np.argwhere(goal_map == 1)
    if len(g_coords) > 0:
        g_r, g_c = g_coords[0]
        goal_v_idx = get_nearest_valid_vertex(g_r, g_c, v_ids)
    else:
        # 목표가 없으면 랜덤 생성 (Warcraft 등)
        print("Warning: 목표 지점이 없습니다. 랜덤 생성합니다.")
        # 중앙 근처나 랜덤한 유효 위치 선택
        valid_indices = np.argwhere(map_design == 1)
        if len(valid_indices) > 0:
            rand_idx = np.random.randint(len(valid_indices))
            g_r, g_c = valid_indices[rand_idx]
            goal_v_idx = get_nearest_valid_vertex(g_r, g_c, v_ids)
            # goal_map 업데이트 (시각화용)
            goal_map[g_r, g_c] = 1
            g_coords = [[g_r, g_c]] # 시각화 로직을 위해 업데이트
        else:
            print("Error: 유효한 공간이 없습니다.")
            goal_v_idx = -1

    # 3. Solver 실행
    geo_dist_map = np.full((rows, cols), np.nan) # 초기화
    
    if goal_v_idx != -1:
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        dists = solver.compute_distance(goal_v_idx)
        
        # 1D 결과 -> 2D 맵 복원
        valid_mask = v_ids != -1
        geo_dist_map[valid_mask] = dists[v_ids[valid_mask]]

    # ==========================================
    # B. Obstacle Distance Field (Euclidean)
    # ==========================================
    print("Computing Obstacle Distance Field (EDT)...")
    
    # scipy.ndimage.distance_transform_edt는 0이 아닌 값에서 0까지의 거리를 계산함
    # map_design은 1=빈공간, 0=장애물
    # 따라서 edt(map_design)을 하면 "빈 공간 픽셀에서 가장 가까운 장애물(0)까지의 거리"가 나옴
    obs_dist_map = distance_transform_edt(map_design)

    # 출발점 생성
    start_node = None
    if args.show_start and opt_dists is not None:
        opt_dist_map = opt_dists[sample_idx].squeeze()
        start_node = get_random_start_node(opt_dist_map)
        if start_node:
            print(f"Generated Start Node: {start_node}")
        else:
            print("Failed to generate a valid start node.")

    # ==========================================
    # C. 시각화 (Visualization)
    # ==========================================
    plt.figure(figsize=(15, 5))

    # 1. Original Map with Goal
    plt.subplot(1, 3, 1)
    plt.title("Occupancy Map & Goal")
    plt.imshow(map_design, cmap='gray')
    if len(g_coords) > 0:
        # g_coords가 list of lists일 수도 있고 numpy array일 수도 있음
        if isinstance(g_coords, list):
             g_r, g_c = g_coords[0]
        else:
             g_r, g_c = g_coords[0]
        plt.scatter(g_c, g_r, c='red', s=100, label='Goal', edgecolors='white')
    
    if start_node:
        s_r, s_c = start_node
        plt.scatter(s_c, s_r, c='green', s=100, label='Start', edgecolors='white')

    plt.legend()

    # 2. Geodesic Distance Field
    plt.subplot(1, 3, 2)
    plt.title("Geodesic Distance Field\n(Heat Method)")
    # 장애물(NaN)은 검은색 처리
    cmap_geo = plt.cm.jet
    cmap_geo.set_bad(color='black')
    plt.imshow(geo_dist_map, cmap=cmap_geo)
    plt.colorbar(label="Distance to Goal")

    # 3. Obstacle Distance Field
    plt.subplot(1, 3, 3)
    plt.title("Obstacle Distance Field\n(Clearance)")
    plt.imshow(obs_dist_map, cmap='viridis') # 장애물과 멀수록 밝음
    plt.colorbar(label="Distance to Nearest Obstacle")

    plt.tight_layout()
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()

# Example usage:
# python heat_method/test.py --mode maze --data_path data/maze/gaps_and_forest_032_moore_c8.npz

# python heat_method/test.py --mode warcraft --warcraft_dir data/warcraft_shortest_path/12x12
# python heat_method/test.py --mode warcraft --warcraft_dir data/warcraft_shortest_path/18x18
# python heat_method/test.py --mode warcraft --warcraft_dir data/warcraft_shortest_path/24x24