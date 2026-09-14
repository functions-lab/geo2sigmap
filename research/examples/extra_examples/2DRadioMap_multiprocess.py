# import numpy as np
# from shapely.geometry import Polygon, box
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from tqdm import tqdm
# import argparse
# import os
# import trimesh

# # ---------------------------
# # Triangle processing function
# # ---------------------------

# def process_triangle_batch(indices, faces, vertices, triangle_values, x_coords, y_coords, cell_size):
#     updates = []
#     for i in indices:
#         face = faces[i]
#         tri_3d = vertices[face]
#         tri_2d = [(x, y) for x, y, z in tri_3d]
#         poly = Polygon(tri_2d)

#         if not (poly.is_valid and poly.area > 0):
#             continue

#         tri_color = triangle_values[i]
#         bounds = poly.bounds

#         x_start = np.searchsorted(x_coords, bounds[0], side='right') - 1
#         x_end   = np.searchsorted(x_coords, bounds[2], side='left')
#         y_start = np.searchsorted(y_coords, bounds[1], side='right') - 1
#         y_end   = np.searchsorted(y_coords, bounds[3], side='left')

#         for xi in range(x_start, x_end):
#             for yi in range(y_start, y_end):
#                 cell = box(
#                     x_coords[xi], y_coords[yi],
#                     x_coords[xi] + cell_size, y_coords[yi] + cell_size
#                 )
#                 area = poly.intersection(cell).area
#                 if area > 1e-2:
#                     updates.append((yi, xi, area * tri_color, area))
#     return updates

# # ---------------------------
# # Main CLI function
# # ---------------------------
# def main(ply_file, color_file, cell_size, n_jobs, output_file):
#     mesh = trimesh.load(ply_file, process=False)
#     vertices = mesh.vertices
#     faces = mesh.faces
#     triangle_values = np.load(color_file)

#     min_x, min_y = mesh.bounds[0][:2]
#     max_x, max_y = mesh.bounds[1][:2]
#     x_coords = np.arange(min_x, max_x, cell_size)
#     y_coords = np.arange(min_y, max_y, cell_size)
#     grid_w, grid_h = len(x_coords), len(y_coords)

#     color_grid = np.zeros((grid_h, grid_w))
#     weight_grid = np.zeros((grid_h, grid_w))

#     print(f"Rasterizing {len(faces)} triangles to {grid_w}x{grid_h} grid...")

#     # args_list = [
#     #     (i, faces, vertices, triangle_values, x_coords, y_coords, cell_size)
#     #     for i in range(len(faces))
#     # ]

#     # Create batches of triangle indices
#     batch_size = 100  # Tune this
#     batches = [list(range(i, min(i + batch_size, len(faces))))
#                for i in range(0, len(faces), batch_size)]
    
#     # Submit batches instead of single triangles
#     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         futures = [
#             executor.submit(process_triangle_batch, batch, faces, vertices, triangle_values,
#                             x_coords, y_coords, cell_size)
#             for batch in batches
#         ]
#         for f in tqdm(as_completed(futures), total=len(futures)):
#             result = f.result()
#             if result is None:
#                 continue
#             for yi, xi, val, area in result:
#                 color_grid[yi, xi] += val
#                 weight_grid[yi, xi] += area

            

#     # with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#     #     futures = [executor.submit(process_triangle, *args) for args in args_list]
#     #     for f in tqdm(as_completed(futures), total=len(futures)):
#     #         updates = f.result()
#     #         if updates is None:
#     #             continue
#     #         for yi, xi, val, area in updates:
#     #             color_grid[yi, xi] += val
#     #             weight_grid[yi, xi] += area

#     final_grid = np.where(weight_grid > 0, color_grid / weight_grid, np.nan)
#     np.save(output_file, final_grid)
#     print(f"Saved output to {output_file}.npy")



            
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ply_file", type=str, required=True, help="Path to input .ply mesh file")
#     parser.add_argument("--color_file", type=str, required=True, help="Path to triangle color .npy file")
#     parser.add_argument("--cell_size", type=float, default=1.0, help="Cell size of the grid")
#     parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel workers")
#     parser.add_argument("--output_file", type=str, default="raster_output", help="Base name for output .npy file")
#     args = parser.parse_args()

#     main(args.ply_file, args.color_file, args.cell_size, args.n_jobs, args.output_file)

import numpy as np
from shapely.geometry import Polygon, box
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from tqdm import tqdm
import argparse
import trimesh
import multiprocessing as mp
import time
# Global references to shared arrays (per worker)
faces = None
vertices = None
triangle_values = None
x_coords_data = None
y_coords_data = None

# Global shared memory references
shm_faces = None
shm_verts = None
shm_values = None
shm_x_coords = None
shm_y_coords = None
g_faces_shape = None
g_verts_shape = None
g_values_shape = None
g_x_coords_shape =None
g_y_coords_shape = None
def init_shared_ndarrays(faces_name, verts_name, values_name, x_coords_name, y_coords_name,
                         faces_shape, verts_shape, values_shape,x_coords_shape, y_coords_shape):
    #print(f"[Init] Running with start method: {mp.get_start_method()}")
    global faces, vertices, triangle_values, x_coords_data, y_coords_data
    # global shm_faces, shm_verts, shm_values, shm_x_coords, shm_y_coords
    # global g_faces_shape, g_verts_shape, g_values_shape, g_x_coords_shape, g_y_coords_shape

    # g_faces_shape = tuple(faces_shape)
    # g_verts_shape = tuple(verts_shape)
    # g_values_shape = tuple(values_shape)
    # g_x_coords_shape = tuple(x_coords_shape)
    # g_y_coords_shape = tuple(y_coords_shape)

    shm_faces = shared_memory.SharedMemory(name=faces_name)
    shm_verts = shared_memory.SharedMemory(name=verts_name)
    shm_values  = shared_memory.SharedMemory(name=values_name)
    shm_x_coords = shared_memory.SharedMemory(name=x_coords_name)
    shm_y_coords = shared_memory.SharedMemory(name=y_coords_name)
    


    # mesh = trimesh.load(ply_file, process=False)
    # vertices = mesh.vertices[:,0:2]
    # faces = mesh.faces
    # triangle_values = np.load(color_file)

    tmp_faces = np.ndarray(faces_shape, dtype=np.int64, buffer=shm_faces.buf)
    faces = tmp_faces.copy()
    
    tmp_vertices = np.ndarray(verts_shape, dtype=np.float64, buffer=shm_verts.buf)
    vertices = tmp_vertices.copy()
    
    tmp_triangle_values = np.ndarray(values_shape, dtype=np.float32, buffer=shm_values.buf)
    triangle_values = tmp_triangle_values.copy()
    

    tmp_x_coords_data  = np.ndarray(x_coords_shape, dtype=np.float64, buffer=shm_x_coords.buf)
    x_coords_data = tmp_x_coords_data.copy()


    tmp_y_coords_data  = np.ndarray(y_coords_shape, dtype=np.float64, buffer=shm_y_coords.buf)
    y_coords_data = tmp_y_coords_data.copy()
    
    
def process_batch(batch_indices, cell_size):
    # global shm_faces, shm_verts, shm_values, shm_x_coords, shm_y_coords
    # global g_faces_shape, g_verts_shape, g_values_shape, g_x_coords_shape, g_y_coords_shape

    # faces = np.ndarray(g_faces_shape, dtype=np.int64, buffer=shm_faces.buf)
    # vertices = np.ndarray(g_verts_shape, dtype=np.float64, buffer=shm_verts.buf)
    # triangle_values = np.ndarray(g_values_shape, dtype=np.float32, buffer=shm_values.buf)
    # x_coords_data = np.ndarray(g_x_coords_shape, dtype=np.float64, buffer=shm_x_coords.buf)
    # y_coords_data = np.ndarray(g_y_coords_shape, dtype=np.float64, buffer=shm_y_coords.buf)
    updates = []
    x_coords = x_coords_data
    y_coords = y_coords_data
    try:
        for i in batch_indices:
            # print(faces.dtype, faces.shape)
            # print(faces[0])
            face = faces[i]
            # print("here", i)
            # print(vertices.dtype, vertices.shape)
            # tri_3d = vertices[face]
            
            # tri_2d = [(x, y) for x, y, z in tri_3d]
            tri_2d = vertices[face]
            poly = Polygon(tri_2d)
    
            if not (poly.is_valid and poly.area > 0):
                continue
    
            tri_color = triangle_values[i]
            bounds = poly.bounds
            
    
            x_start = np.searchsorted(x_coords, bounds[0], side='right') - 1
            x_end   = np.searchsorted(x_coords, bounds[2], side='left')
            y_start = np.searchsorted(y_coords, bounds[1], side='right') - 1
            y_end   = np.searchsorted(y_coords, bounds[3], side='left')
    
            for xi in range(x_start, x_end):
                for yi in range(y_start, y_end):
                    cell = box(
                        x_coords[xi], y_coords[yi],
                        x_coords[xi] + cell_size, y_coords[yi] + cell_size
                    )
                    if not poly.intersects(cell):
                        continue  # Skip early
                    area = poly.intersection(cell).area
                    if area > 1e-2:
                        updates.append((yi, xi, area * tri_color, area))
    except Exception as e:
        print(f"[ERROR] In batch {batch_indices[0]}–{batch_indices[-1]}: {e}")
        return []  # or raise to debug

    return updates

def main(ply_file, color_file, cell_size, n_jobs, batch_size, output_file):
    mesh = trimesh.load(ply_file, process=False)
    verts = mesh.vertices[:,0:2]
    faces_data = mesh.faces
    print("faces_data.shape",faces_data.shape)
    
    triangle_values_data = np.load(color_file)
    print("verts.shape",verts.shape)

    min_x, min_y = mesh.bounds[0][:2]
    max_x, max_y = mesh.bounds[1][:2]
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    grid_w, grid_h = len(x_coords), len(y_coords)

    color_grid = np.zeros((grid_h, grid_w))
    weight_grid = np.zeros((grid_h, grid_w))

    # Shared memory allocation
    shm_faces = shared_memory.SharedMemory(create=True, size=faces_data.nbytes)
    shm_verts = shared_memory.SharedMemory(create=True, size=verts.nbytes)
    shm_vals  = shared_memory.SharedMemory(create=True, size=triangle_values_data.nbytes)
    shm_x_coords = shared_memory.SharedMemory(create=True, size=x_coords.nbytes)
    shm_y_coords = shared_memory.SharedMemory(create=True, size=y_coords.nbytes)
    print("x_coords.dtype",x_coords.dtype)
    
 
    # Assume x_coords and y_coords are 1D sorted arrays from np.arange
    # xv, yv = np.meshgrid(x_coords, y_coords)
    
    # xv = xv.ravel()
    # yv = yv.ravel()
    
    # # Vectorized box creation for each grid cell
    # cell_boxes = [
    #     box(x, y, x + cell_size, y + cell_size)
    #     for x, y in zip(xv, yv)
    # ]


    np.copyto(np.ndarray(faces_data.shape, dtype=faces_data.dtype, buffer=shm_faces.buf), faces_data)
    np.copyto(np.ndarray(verts.shape, dtype=verts.dtype, buffer=shm_verts.buf), verts)

    np.copyto(np.ndarray(triangle_values_data.shape, dtype=triangle_values_data.dtype, buffer=shm_vals.buf), triangle_values_data)

    np.copyto(np.ndarray(y_coords.shape, dtype=y_coords.dtype, buffer=shm_y_coords.buf), y_coords)
    np.copyto(np.ndarray(x_coords.shape, dtype=x_coords.dtype, buffer=shm_x_coords.buf), x_coords)

    face_shape = faces_data.shape
    vert_shape = verts.shape
    vals_shape = triangle_values_data.shape
    x_coords_shape = x_coords.shape
    y_coords_shape = y_coords.shape

    batches = [list(range(i, min(i + batch_size, len(faces_data))))
               for i in range(0, len(faces_data), batch_size)]

    print(f"Processing {len(faces_data)} triangles in {len(batches)} batches using {n_jobs} workers...")

    ctx = mp.get_context("spawn")

   
                               
    with ProcessPoolExecutor(
        mp_context=ctx,
        max_workers=n_jobs,
        initializer=init_shared_ndarrays,
        initargs=(shm_faces.name, shm_verts.name, shm_vals.name,shm_x_coords.name, shm_y_coords.name, 
                  face_shape, vert_shape, vals_shape, x_coords_shape, y_coords_shape)
    ) as executor:
        futures = [executor.submit(process_batch, batch, cell_size)
                   for batch in batches]

        for f in tqdm(as_completed(futures), total=len(futures)):
            result = f.result()
            if result:
                for yi, xi, value, area in result:
                    color_grid[yi, xi] += value
                    weight_grid[yi, xi] += area

    shm_faces.close(); shm_faces.unlink()
    shm_verts.close(); shm_verts.unlink()
    shm_vals.close(); shm_vals.unlink()

    final_grid = np.where(weight_grid > 0, color_grid / weight_grid, np.nan)
    np.save(output_file, final_grid)
    print(f"Output saved to {output_file}.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_file", type=str, required=True)
    parser.add_argument("--color_file", type=str, required=True)
    parser.add_argument("--cell_size", type=float, default=1.0)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--output_file", type=str, default="output_raster")
    args = parser.parse_args()

    main(args.ply_file, args.color_file, args.cell_size,
         args.n_jobs, args.batch_size, args.output_file)

