import glob
import os
import csv
import numpy as np
import torch
import torch.nn.functional as thf
import h5py
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pdb
from tqdm import tqdm, trange

hypersim_path = "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads/"
csv_filename = os.path.join(hypersim_path, "metadata_images_split_scene_v1.csv")
assert(os.path.exists(csv_filename))

# read the csv file first
with open(csv_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    metadata = {}
    for row in reader:
        for column, value in row.items():
            metadata.setdefault(column, []).append(value)

# not only train
included_in_public_release = np.array(metadata["included_in_public_release"])
public_index = included_in_public_release=='True'
split_partitions = np.array(metadata["split_partition_name"])[public_index]
scene_names = np.array(metadata["scene_name"])[public_index]
camera_names = np.array(metadata["camera_name"])[public_index]
frame_ids = np.array(metadata["frame_id"])[public_index].astype(np.int64)
scene_name_unique = np.unique(scene_names)

# read cameras
camera_filename = os.path.join(hypersim_path, "metadata_camera_parameters.csv")
assert(os.path.exists(camera_filename))
with open(camera_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    camera_metadata = {}
    for row in reader:
        for column, value in row.items():
            camera_metadata.setdefault(column, []).append(value)


def gen_cam_uv(wid, hei):
    u_min  = -1.0
    u_max  = 1.0
    v_min  = -1.0
    v_max  = 1.0
    half_du = 0.5 * (u_max - u_min) / wid
    half_dv = 0.5 * (v_max - v_min) / hei

    u, v = np.meshgrid(np.linspace(u_min+half_du, u_max-half_du, wid),
                       np.linspace(v_min+half_dv, v_max-half_dv, hei)[::-1])
    uvs_2d = np.dstack((u,v,np.ones_like(u)))
    return torch.tensor(uvs_2d).cuda().float()


def save_ply(filename, points):
    """
    Save 3D points to a .ply file.
    
    Parameters:
    - filename: String. The name of the file where to save the point cloud.
    - points: NumPy array of shape (n_points, 3). The array containing the point cloud, where each row is a point (x, y, z).
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    np.savetxt(filename, points, fmt="%.6f %.6f %.6f", header=header, comments='')


csv_file = open("multi_view_status.csv", 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["scene_name","camera_name","frame_id",'split', 'nan_ratio', 'neighbor_views', 'overlaps', 'num_overlaps', 'max_overlaps'])
for scene_name in tqdm(scene_name_unique):
    scene_mask = scene_names == scene_name
    camera_names_perscene = camera_names[scene_mask]
    frame_ids_perscene = frame_ids[scene_mask]
    split_perscene = split_partitions[scene_mask]

    i_ = camera_metadata["scene_name"].index(scene_name)
    width_pixels = int(np.round(float(camera_metadata["settings_output_img_width"][i_])))
    height_pixels = int(np.round(float(camera_metadata["settings_output_img_height"][i_])))
    meters_per_asset_unit = float(camera_metadata["settings_units_info_meters_scale"][i_])
    key_list = ["M_cam_from_uv_00", "M_cam_from_uv_01", "M_cam_from_uv_02",
               "M_cam_from_uv_10", "M_cam_from_uv_11", "M_cam_from_uv_12",
               "M_cam_from_uv_20", "M_cam_from_uv_21", "M_cam_from_uv_22"]
    uv2c = np.array([
        camera_metadata[n_][i_] for n_ in key_list
    ]).astype(np.float32).reshape(3, 3)

    # key_list = ["M_proj_00", "M_proj_01", "M_proj_02", "M_proj_03",
    #             "M_proj_10", "M_proj_11", "M_proj_12", "M_proj_13",
    #             "M_proj_20", "M_proj_21", "M_proj_22", "M_proj_23",
    #             "M_proj_30", "M_proj_31", "M_proj_32", "M_proj_33",]
    # M_proj = np.array([
    #     camera_metadata[n_][i_] for n_ in key_list
    # ]).astype(np.float32).reshape(4, 4)
    
    c2ws = []
    depths = []
    cam_frame_split = []
    for cam_name in np.unique(camera_names_perscene):
        camera_pos_hdf5 = os.path.join(hypersim_path, scene_name, "_detail", cam_name, "camera_keyframe_positions.hdf5")
        camera_c2w_hdf5 = os.path.join(hypersim_path, scene_name, "_detail", cam_name, "camera_keyframe_orientations.hdf5")

        with h5py.File(camera_pos_hdf5, "r") as f: camera_poss = f["dataset"][:]
        with h5py.File(camera_c2w_hdf5, "r") as f: camera_c2ws = f["dataset"][:]
        
        cam_mask = camera_names_perscene == cam_name
        frame_ids_percam = frame_ids_perscene[cam_mask]
        split_percam = split_perscene[cam_mask]
        for frame_i, split_i in zip(frame_ids_percam, split_percam):
            depths_meters_hdf5 = os.path.join(hypersim_path, scene_name, "images", f"scene_{cam_name}_geometry_hdf5", f"frame.{int(frame_i):04d}.depth_meters.hdf5")
            with h5py.File(depths_meters_hdf5, "r") as f: hypersim_depth_meters = f["dataset"][:].astype(np.float32)

            camera_pos = camera_poss[frame_i]
            camera_c2w = camera_c2ws[frame_i]
            extrin = np.eye(4)
            extrin[:3, :3] = camera_c2w
            extrin[:3, 3] = camera_pos * meters_per_asset_unit
            depths.append(hypersim_depth_meters)
            c2ws.append(extrin)
            cam_frame_split.append((cam_name, frame_i, split_i))

    uv2c = torch.tensor(uv2c).float().cuda()
    depths = [torch.tensor(d).float().cuda() for d in depths]
    c2ws = [torch.tensor(c).float().cuda() for c in c2ws]
    num_frames = len(cam_frame_split)
    for i in trange(num_frames, leave=False):
        depth_i = depths[i]
        c2w_i = c2ws[i]

        # nan_ratio
        nan_ma = torch.isnan(depth_i)
        nan_ratio = torch.count_nonzero(nan_ma).item() / (width_pixels * height_pixels)

        # project to world
        uv = gen_cam_uv(width_pixels, height_pixels).reshape(-1, 3)
        pt3d = uv2c @ uv.T
        pt3d = pt3d / torch.norm(pt3d, dim=0, keepdim=True)
        pt3d = pt3d * depth_i.reshape(1, -1)
        pt3d = c2w_i[:3, :3] @ pt3d + c2w_i[:3, 3:4]

        overlap_ratios = []
        for j in range(num_frames):
            if i == j:
                overlap_ratios.append(-1)
                continue

            # compute overlap
            depth_j = depths[j]
            c2w_j = c2ws[j]

            # project to camera
            pt3d_j = pt3d - c2w_j[:3, 3:4]
            pt3d_j = torch.inverse(c2w_j[:3, :3]) @ pt3d_j
            uv_j = torch.inverse(uv2c) @ pt3d_j
            uv_j = uv_j[:2] / uv_j[2:]
            depth_ij = torch.norm(pt3d_j, dim=0, keepdim=True)

            # grap depth from view
            grid = uv_j.reshape(2, height_pixels, width_pixels)
            grid[1] = -grid[1]
            grid = grid.permute(1, 2, 0)[None]
            depth_reproj = thf.grid_sample(depth_j[None, None], grid, mode='nearest', padding_mode="zeros", align_corners=True)

            depth_reproj = depth_reproj[0, 0]
            depth_ij = depth_ij.reshape(height_pixels, width_pixels)
            uv_j = uv_j.reshape(2, height_pixels, width_pixels)
            
            visible_mask = torch.logical_and(uv_j > -1, uv_j < 1)
            visible_mask = torch.logical_and(visible_mask[:1], visible_mask[1:])
            visible_mask = torch.logical_and(depth_ij < depth_reproj + 0.05, visible_mask)
            visible_mask = torch.logical_and(depth_ij > depth_reproj - 0.05, visible_mask)
            overlap_ratio = torch.count_nonzero(visible_mask) / (height_pixels * width_pixels)
            overlap_ratios.append(overlap_ratio.item())

        overlap_order = np.argsort(overlap_ratios)
        overlap_ratios_sorted = [overlap_ratios[i_] for i_ in overlap_order]
        cam_frame_sorted = [cam_frame_split[i_] for i_ in overlap_order]

        ith = np.searchsorted(overlap_ratios_sorted, 0.4)
        overlap_ratios_sorted = overlap_ratios_sorted[ith:][::-1]
        cam_frame_sorted = cam_frame_sorted[ith:][::-1]
        
        max_overlap_ratio = overlap_ratios_sorted[0] if len(overlap_ratios_sorted) > 0 else 0
        overlap_ratios_str = [f"{n:.3f}" for n in overlap_ratios_sorted]
        cam_frame_str = [c + '|' + str(f) for c, f, s in cam_frame_sorted]

        camera_name, frame_id, split = cam_frame_split[i]
        csv_writer.writerow([
            # "scene_name","camera_name","frame_id",'split', 'nan_ratio', 'neighbor_views', 'overlaps', 'num_overlap', 'max_overlaps'
            scene_name, camera_name, frame_id, split,
            f"{nan_ratio:.5}",
            '&'.join(cam_frame_str), '&'.join(overlap_ratios_str),
            len(overlap_ratios_str), max_overlap_ratio
        ])

csv_file.close()