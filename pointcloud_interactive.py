import numpy as np
import open3d as o3d
from skimage import measure
from scipy.spatial import cKDTree


def load_point_cloud(file_path):
    print(f"[INFO] Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"[INFO] Loaded point cloud with {np.asarray(pcd.points).shape[0]} points.")
    return pcd


def clean_point_cloud(pcd, std_ratio=2.0, nb_neighbors=20, verbose=True):
    """
    Clean the point cloud using statistical outlier removal.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        std_ratio (float): Higher means less aggressive filtering.
        nb_neighbors (int): Number of neighbors to analyze.
        verbose (bool): Whether to print info.
    
    Returns:
        o3d.geometry.PointCloud: Cleaned point cloud.
    """
    if verbose:
        print(f"[INFO] Cleaning point cloud... std_ratio={std_ratio}, nb_neighbors={nb_neighbors}")
        print(f"[INFO] Before cleaning: {len(pcd.points)} points")
    
    cleaned_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    if verbose:
        print(f"[INFO] After cleaning: {len(cleaned_pcd.points)} points (removed {len(pcd.points) - len(cleaned_pcd.points)})")

    return cleaned_pcd


def preprocess_normals(pcd, radius=0.05, max_nn=50):
    print("[INFO] Estimating and orienting normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_consistent_tangent_plane(k=30)
    return pcd


def poisson_mesh(pcd, depth=12, density_percentile=5):
    print("[INFO] Applying Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    densities = np.asarray(densities)
    density_thresh = np.percentile(densities, density_percentile)
    mesh.remove_vertices_by_mask(densities < density_thresh)

    # Optional: crop to bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)

    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()

    print(f"[INFO] Final Poisson mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    return mesh


def marching_cubes_mesh(pcd, voxel_size=0.01, iso_level_percentile=5):
    print("[INFO] Converting point cloud to mesh using Marching Cubes...")
    points = np.asarray(pcd.points)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    x = np.arange(mins[0], maxs[0], voxel_size)
    y = np.arange(mins[1], maxs[1], voxel_size)
    z = np.arange(mins[2], maxs[2], voxel_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    tree = cKDTree(points)
    distances, _ = tree.query(grid_points)
    scalar_field = distances.reshape(X.shape)

    iso_level = np.percentile(distances, iso_level_percentile)
    print(f"[INFO] Using iso-level: {iso_level:.4f}")

    verts, faces, normals, _ = measure.marching_cubes(scalar_field, level=iso_level)

    verts_world = np.zeros_like(verts)
    verts_world[:, 0] = x[0] + verts[:, 0] * voxel_size
    verts_world[:, 1] = y[0] + verts[:, 1] * voxel_size
    verts_world[:, 2] = z[0] + verts[:, 2] * voxel_size

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    print(f"[INFO] Marching Cubes mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces.")
    return mesh


def view_point_cloud(pcd):
    print("[INFO] Visualizing point cloud...")
    o3d.visualization.draw_geometries([pcd])


def view_mesh(mesh):
    print("[INFO] Visualizing mesh...")
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    # === SETTINGS ===
    point_cloud_file = "point_cloudm.ply"
    use_poisson = True  # Set False to use Marching Cubes instead
    voxel_size = 0.005

    # === Load and preprocess ===
    pcd = load_point_cloud(point_cloud_file)
    pcd = clean_point_cloud(pcd, std_ratio=2.4)
    pcd = preprocess_normals(pcd, radius=0.05)

    view_point_cloud(pcd)

    # === Mesh generation ===
    if use_poisson:
        mesh = poisson_mesh(pcd, depth=13, density_percentile=5)
    else:
        mesh = marching_cubes_mesh(pcd, voxel_size=voxel_size, iso_level_percentile=5)

    view_mesh(mesh)
