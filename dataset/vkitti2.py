import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor


class VKITTI2(Dataset):
    def __init__(self, data_dir_root, do_kb_crop=True, split="test", preprocess=None):
        import glob
        data_dir_root = os.path.abspath(data_dir_root)
        # image paths are of the form <data_dir_root>/rgb/<scene>/<variant>/frames/<rgb,depth>/Camera<0,1>/rgb_{}.jpg
        self.image_files = sorted(glob.glob(os.path.join(
            data_dir_root, "**", "frames", "rgb", "Camera_0", '*.jpg'), recursive=True))
        self.depth_files = [r.replace("/rgb/", "/depth/").replace(
            "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        self.do_kb_crop = do_kb_crop

        # If train test split is not created, then create one.
        # Split is such that 8% of the frames from each scene are used for testing.
        if not os.path.exists(os.path.join(data_dir_root, "train.txt")):
            import random
            scenes = set([os.path.basename(os.path.dirname(
                os.path.dirname(os.path.dirname(f)))) for f in self.image_files])
            train_files = []
            test_files = []
            for scene in scenes:
                scene_files = [f for f in self.image_files if os.path.basename(
                    os.path.dirname(os.path.dirname(os.path.dirname(f)))) == scene]
                random.shuffle(scene_files)
                train_files.extend(scene_files[:int(len(scene_files) * 0.92)])
                test_files.extend(scene_files[int(len(scene_files) * 0.92):])
            with open(os.path.join(data_dir_root, "train.txt"), "w") as f:
                f.write("\n".join(train_files))
            with open(os.path.join(data_dir_root, "test.txt"), "w") as f:
                f.write("\n".join(test_files))

        if split == "train":
            with open(os.path.join(data_dir_root, "train.txt"), "r") as f:
                self.image_files = f.read().splitlines()
            self.image_files = [r.replace("data/VKITTI2", data_dir_root) for r in self.image_files]
            self.depth_files = [r.replace("/rgb/", "/depth/").replace(
                "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        elif split == "test":
            with open(os.path.join(data_dir_root, "test.txt"), "r") as f:
                self.image_files = f.read().splitlines()
            self.image_files = [r.replace("data/VKITTI2", data_dir_root) for r in self.image_files]
            self.depth_files = [r.replace("/rgb/", "/depth/").replace(
                "rgb_", "depth_").replace(".jpg", ".png") for r in self.image_files]
        else:
            raise RuntimeError(f"VKITTI2: split = {split} unrecognized")
        
        self.preprocess = preprocess if preprocess is not None else lambda x: x

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH) / 100.0  # cm to m
        depth = Image.fromarray(depth)

        if self.do_kb_crop:
            if idx == 0:
                print("Using KB input crop")
            height = image.height
            width = image.width
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            depth = depth.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            image = image.crop(
                (left_margin, top_margin, left_margin + 1216, top_margin + 352))
            # uv = uv[:, top_margin:top_margin + 352, left_margin:left_margin + 1216]

        image = ToTensor()(image)
        image = 2 * image -1
        depth = np.asarray(depth, dtype=np.float32) / 1.
        depth = torch.tensor(depth)[None, ...]
        sample = dict(image=image, depth=depth, dataset='kitti')
        return self.preprocess(sample)

    def __len__(self):
        return len(self.image_files)


def get_vkitti2_loader(data_dir_root, batch_size=1, **kwargs):
    dataset = VKITTI2(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)


if __name__ == "__main__":
    loader = get_vkitti2_loader(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/nerf_public/VKITTI2")
    print("Total files", len(loader.dataset))
    for i, sample in enumerate(loader):
        print(sample["image"].shape)
        print(sample["depth"].shape)
        print(sample["dataset"])
        print(sample['depth'].min(), sample['depth'].max())
        if i > 5:
            break