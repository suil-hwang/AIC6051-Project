"""Distance Field computation utilities
"""
import numpy as np
import torch
import potpourri3d as pp3d
from scipy.ndimage import distance_transform_edt
from typing import Tuple, Optional


def array_to_mesh(binary_map_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D binary map을 mesh로 변환 (1: passable, 0: obstacle)"""
    rows, cols = binary_map_2d.shape
    vertex_ids = np.full((rows, cols), -1, dtype=int)
    vertices = []
    curr_id = 0
    
    # Create vertices for passable cells
    for r in range(rows):
        for c in range(cols):
            if binary_map_2d[r, c] == 1:
                vertex_ids[r, c] = curr_id
                vertices.append([float(r), float(c), 0.0])
                curr_id += 1
    
    vertices = np.array(vertices, dtype=np.float64)
    
    # Create triangular faces
    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v_tl = vertex_ids[r, c]
            v_tr = vertex_ids[r, c+1]
            v_bl = vertex_ids[r+1, c]
            v_br = vertex_ids[r+1, c+1]
            
            if v_tl != -1 and v_bl != -1 and v_tr != -1:
                faces.append([v_tl, v_bl, v_tr])
            if v_tr != -1 and v_bl != -1 and v_br != -1:
                faces.append([v_tr, v_bl, v_br])
    
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces, vertex_ids


def remove_unreferenced_vertices(
    vertices: np.ndarray, 
    faces: np.ndarray, 
    vertex_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """고립된 vertex 제거"""
    referenced_vertices = np.unique(faces.flatten())
    new_vertices = vertices[referenced_vertices]
    
    old_to_new = np.full(len(vertices), -1, dtype=int)
    old_to_new[referenced_vertices] = np.arange(len(referenced_vertices))
    
    new_faces = old_to_new[faces]
    new_vertex_ids = np.where(vertex_ids != -1, old_to_new[vertex_ids], -1)
    
    return new_vertices, new_faces, new_vertex_ids


def compute_geodesic_distance_field(
    map_design: np.ndarray,
    goal_map: np.ndarray,
    fallback_value: float = 100.0
) -> np.ndarray:
    """
    Heat Method를 사용한 Geodesic Distance Field 계산
    
    Args:
        map_design: (H, W) binary map (1: passable, 0: obstacle)
        goal_map: (H, W) one-hot goal location
        fallback_value: mesh 생성 실패 시 사용할 기본값
        
    Returns:
        (H, W) geodesic distance field
    """
    rows, cols = map_design.shape
    
    try:
        # Convert to mesh
        V, F, v_ids = array_to_mesh(map_design)
        
        if len(F) == 0:
            print("Warning: Empty mesh, using fallback")
            return np.full((rows, cols), fallback_value)
        
        # Clean mesh
        V, F, v_ids = remove_unreferenced_vertices(V, F, v_ids)
        
        # Find goal vertex
        g_coords = np.argwhere(goal_map == 1)
        if len(g_coords) == 0:
            print("Warning: No goal found, using fallback")
            return np.full((rows, cols), fallback_value)
        
        g_r, g_c = g_coords[0]
        goal_v_idx = v_ids[g_r, g_c]
        
        if goal_v_idx == -1:
            # Find nearest valid vertex
            valid_indices = np.argwhere(v_ids != -1)
            if len(valid_indices) == 0:
                return np.full((rows, cols), fallback_value)
            
            dists = (valid_indices[:, 0] - g_r)**2 + (valid_indices[:, 1] - g_c)**2
            nearest_idx = np.argmin(dists)
            g_r, g_c = valid_indices[nearest_idx]
            goal_v_idx = v_ids[g_r, g_c]
        
        # Compute geodesic distance
        solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        dists = solver.compute_distance(goal_v_idx)
        
        # Map back to 2D grid
        geo_dist_map = np.full((rows, cols), fallback_value)
        valid_mask = v_ids != -1
        geo_dist_map[valid_mask] = dists[v_ids[valid_mask]]
        
        return geo_dist_map
        
    except Exception as e:
        print(f"Warning: Geodesic computation failed ({e}), using fallback")
        return np.full((rows, cols), fallback_value)


def compute_obstacle_distance_field(map_design: np.ndarray) -> np.ndarray:
    """
    Euclidean Distance Transform를 사용한 Obstacle Distance Field 계산
    
    Args:
        map_design: (H, W) binary map (1: passable, 0: obstacle)
        
    Returns:
        (H, W) obstacle distance field
    """
    return distance_transform_edt(map_design).astype(np.float32)


def precompute_distance_fields(
    map_designs: np.ndarray,
    goal_maps: np.ndarray,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    배치 데이터에 대한 distance fields 사전 계산
    
    Args:
        map_designs: (N, 1, H, W) or (N, H, W)
        goal_maps: (N, 1, H, W) or (N, H, W)
        
    Returns:
        geodesic_fields: (N, 1, H, W)
        obstacle_fields: (N, 1, H, W)
    """
    # Squeeze if needed
    if map_designs.ndim == 4:
        map_designs = map_designs[:, 0]
    if goal_maps.ndim == 4:
        goal_maps = goal_maps[:, 0]
    
    N = len(map_designs)
    H, W = map_designs.shape[1:3]
    
    geodesic_fields = np.zeros((N, 1, H, W), dtype=np.float32)
    obstacle_fields = np.zeros((N, 1, H, W), dtype=np.float32)
    
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(range(N), desc="Computing distance fields")
        except ImportError:
            iterator = range(N)
            print("Computing distance fields...")
    else:
        iterator = range(N)
    
    for i in iterator:
        # Geodesic distance
        geodesic_fields[i, 0] = compute_geodesic_distance_field(
            map_designs[i], 
            goal_maps[i]
        )
        
        # Obstacle distance
        obstacle_fields[i, 0] = compute_obstacle_distance_field(map_designs[i])
    
    return geodesic_fields, obstacle_fields


def compute_normalization_constants(
    map_designs: np.ndarray,
    goal_maps: np.ndarray,
) -> Tuple[float, float]:
    """
    데이터셋 전체에서 정규화 상수 계산
    
    Returns:
        d_max_geo: geodesic용 정규화 상수 (맵 대각선)
        d_max_obs: obstacle용 정규화 상수 (전체 최대값)
    """
    H, W = map_designs.shape[-2:]
    
    # Geodesic: 맵 대각선 길이
    d_max_geo = np.sqrt(H**2 + W**2)
    
    # Obstacle: 실제 데이터에서 계산 (첫 100개 샘플로 추정)
    n_samples = min(100, len(map_designs))
    max_obs_dists = []
    
    for i in range(n_samples):
        obs_dist = compute_obstacle_distance_field(map_designs[i].squeeze())
        max_obs_dists.append(obs_dist.max())
    
    d_max_obs = np.percentile(max_obs_dists, 95)  # 95th percentile for robustness
    
    return float(d_max_geo), float(d_max_obs)


def precompute_distance_fields_normalized(
    map_designs: np.ndarray,
    goal_maps: np.ndarray,
    d_max_geo: float = None,
    d_max_obs: float = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    정규화된 distance fields 사전 계산
    
    Returns:
        geodesic_fields_norm: (N, 1, H, W) normalized [0, 1]
        obstacle_fields_norm: (N, 1, H, W) normalized [0, 1]
        d_max_geo: 사용된 geodesic 정규화 상수
        d_max_obs: 사용된 obstacle 정규화 상수
    """
    
    # 정규화 상수 계산 또는 사용
    if d_max_geo is None or d_max_obs is None:
        print("Computing normalization constants...")
        d_max_geo, d_max_obs = compute_normalization_constants(
            map_designs, goal_maps
        )
    
    print(f"Normalization constants: geo={d_max_geo:.2f}, obs={d_max_obs:.2f}")
    
    # Distance fields 계산
    geodesic_fields, obstacle_fields = precompute_distance_fields(
        map_designs, goal_maps, show_progress
    )
    
    # 정규화
    geodesic_fields_norm = geodesic_fields / d_max_geo
    obstacle_fields_norm = obstacle_fields / d_max_obs
    
    # Clip to [0, 1] for safety
    geodesic_fields_norm = np.clip(geodesic_fields_norm, 0, 1)
    obstacle_fields_norm = np.clip(obstacle_fields_norm, 0, 1)
    
    return geodesic_fields_norm, obstacle_fields_norm, d_max_geo, d_max_obs