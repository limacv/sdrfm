import glob
import os

import h5py
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Resize


_NAN_THRESHOLD = 0.02
_OVERLAP_THRESHOLD = 5
_MAXOVERLAP_THRESHOLD = 0.5


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
    return uvs_2d


# my implementation that consider tilt shift
def hypersim_distance_to_depth_batched(npyDistance, uv2c_mat):
    bsz, hei, wid = npyDistance.shape
    uv = gen_cam_uv(wid, hei).reshape(-1, 3)
    pt3d = uv2c_mat @ uv.T
    pt3d = pt3d / np.linalg.norm(pt3d, axis=0, keepdims=True)
    pt3d = torch.tensor(pt3d[:, None]) * npyDistance.reshape(1, bsz, -1)
    return - pt3d[-1].reshape(bsz, hei, wid)


# This one is deprecated because it's slightly wrong 
#   (assume same intrinsic for all image, which is not true)
def hypersim_distance_to_depth_old(npyDistance):
    intWidth, intHeight, fltFocal = 1024, 768, 886.81

    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(
        1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5,
                                 intHeight).reshape(intHeight, 1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)

    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


class HyperSimMultiView(Dataset):
    def __init__(self, topk, data_dir_root, preprocess=None, split='test'):
        self.data_dir_root = data_dir_root
        self.topk = topk
        csv_filename = os.path.join(data_dir_root, "multi_view_status.csv")
        assert(os.path.exists(csv_filename))
        
        # read the csv file first
        with open(csv_filename, encoding="UTF-8") as file:
            reader = csv.DictReader(file)
            metadata = {}
            for row in reader:
                for column, value in row.items():
                    metadata.setdefault(column, []).append(value)
        
        # read the camera intrinsic file
        camera_metadata_filename = os.path.join(data_dir_root, "metadata_camera_parameters.csv")
        with open(camera_metadata_filename, encoding="UTF-8") as file:
            reader = csv.DictReader(file)
            camera_metadata = {}
            for row in reader:
                for column, value in row.items():
                    camera_metadata.setdefault(column, []).append(value)
        # parse the file
        self.uv2c_dict, self.unit2meter_dict = {}, {}
        for i_, scene_name in enumerate(camera_metadata["scene_name"]):
            meters_per_asset_unit = float(camera_metadata["settings_units_info_meters_scale"][i_])
            key_list = ["M_cam_from_uv_00", "M_cam_from_uv_01", "M_cam_from_uv_02",
                "M_cam_from_uv_10", "M_cam_from_uv_11", "M_cam_from_uv_12",
                "M_cam_from_uv_20", "M_cam_from_uv_21", "M_cam_from_uv_22"]
            uv2c = np.array([
                camera_metadata[n_][i_] for n_ in key_list
                ]).astype(np.float32).reshape(3, 3)
            self.uv2c_dict[scene_name] = uv2c
            self.unit2meter_dict[scene_name] = meters_per_asset_unit

        # filter out dirty data
        split_partition = np.array(metadata["split"])
        split_index = split_partition == split
        nan_ratio = np.array(metadata["nan_ratio"]).astype(np.float32)
        filter_mask = nan_ratio < _NAN_THRESHOLD # predefined thre
        filter_mask = np.logical_and(filter_mask, split_index)

        overlap_num = np.array(metadata["num_overlaps"]).astype(np.int64)
        overlap_ma = overlap_num >= _OVERLAP_THRESHOLD
        filter_mask = np.logical_and(filter_mask, overlap_ma)

        max_overlaps = np.array(metadata["max_overlaps"]).astype(np.float32)
        overlap_ma = max_overlaps >= _MAXOVERLAP_THRESHOLD
        filter_mask = np.logical_and(filter_mask, overlap_ma)

        self.scene_names = np.array(metadata["scene_name"])[filter_mask].tolist()
        self.camera_names = np.array(metadata["camera_name"])[filter_mask].tolist()
        self.frame_ids = np.array(metadata["frame_id"])[filter_mask].tolist()
        self.overlap_ratios = np.array(metadata["overlaps"])[filter_mask].tolist()
        self.overlap_views = np.array(metadata["neighbor_views"])[filter_mask].tolist()
        # parse overlap strings
        self.overlap_ratios = [tuple(map(float, s_.split('&'))) if len(s_) > 0 else tuple() for s_ in self.overlap_ratios]
        self.overlap_views = [tuple(map(lambda x: tuple(x.split('|')), 
                                         s_.split('&'))) if len(s_) > 0 else tuple() for s_ in self.overlap_views]

        # self.image_files = sorted(glob.glob(os.path.join(
        #     data_dir_root, 'ai_001_001', 'images', 'scene_cam_*_final_preview', '*.tonemap.jpg')))
        # self.depth_files = [r.replace("_final_preview", "_geometry_hdf5").replace(
        #     ".tonemap.jpg", ".depth_meters.hdf5") for r in self.image_files]
        
        # completeness check
        for scene, cam, frm in zip(self.scene_names, self.camera_names, self.frame_ids):
            p = self._get_image_files(scene, cam, frm)
            assert os.path.isfile(p), f"HyperSim::{p} does not exist"
            p = self._get_depth_files(scene, cam, frm)
            assert os.path.isfile(p), f"HyperSim::{p} does not exist"

        self.preprocess = preprocess if preprocess is not None else lambda x: x

    def _get_image_files(self, scene_name, camera_name, frame_id):
        return os.path.join(
            self.data_dir_root, scene_name, "images", f"scene_{camera_name}_final_preview", f"frame.{int(frame_id):04d}.tonemap.jpg"
        )

    def _get_depth_files(self, scene_name, camera_name, frame_id):
        return os.path.join(
            self.data_dir_root, scene_name, "images", f"scene_{camera_name}_geometry_hdf5", f"frame.{int(frame_id):04d}.depth_meters.hdf5"
        )
    
    def _get_extrinsic(self, scene_name, cam_list, frame_list):
        raise NotImplementedError  # TODO: implement read extrinsic if needed
        
    def __getitem__(self, idx):
        scene, cam, frm = self.scene_names[idx], self.camera_names[idx], self.frame_ids[idx]
        image_path = self._get_image_files(scene, cam, frm)
        depth_path = self._get_depth_files(scene, cam, frm)
        neighbors = self.overlap_views[idx]
        overlap_ratios = self.overlap_ratios[idx]

        neighbors = neighbors[:self.topk]
        overlap_ratios = overlap_ratios[:self.topk]

        image = [Image.open(image_path)]
        depth = [np.array(h5py.File(depth_path, "r")['dataset']).astype(np.float32)]

        for (cam, frm), overlap_ratio in zip(neighbors, overlap_ratios):
            image_path = self._get_image_files(scene, cam, frm)
            depth_path = self._get_depth_files(scene, cam, frm)
            image.append(Image.open(image_path))
            depth.append(np.array(h5py.File(depth_path, "r")['dataset']).astype(np.float32))

        # image
        images = [ToTensor()(img) for img in image]
        images = torch.stack(images)
        images = images * 2 - 1

        depths = torch.tensor(depth)
        depths = hypersim_distance_to_depth_batched(depths, self.uv2c_dict[scene])

        depth_max = depths[np.logical_not(np.isnan(depths))].max()
        depths = np.nan_to_num(depths, nan=depth_max, neginf=depth_max, posinf=depth_max)
        
        depths = torch.tensor(depths)[None]
        sample = dict(images=images, depths=depths, dataset="mv_hypersim")
        return self.preprocess(sample)

    def __len__(self):
        return len(self.scene_names)


def get_multiview_hypersim_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = HyperSimMultiView(5, data_dir_root, split='train')
    return DataLoader(dataset, batch_size, **kwargs)

if __name__ == "__main__":
    # depth nan question: https://github.com/apple/ml-hypersim/issues/13
    # not all pixels contain actual scene geometry, e.g., background pixels
    # In most learning applications, masking them out and ignoring them \
        # when computing a per-pixel loss seems like a sensible approach.
    loader = get_multiview_hypersim_loader(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["images"].shape)
        print(sample["depths"].shape)
        # print(sample["dataset"])
        print(sample['depths'].min(), sample['depths'].max())
        if i > 5:
            break