from torch.utils.data import Dataset
from .hypersim import HyperSim
from .vkitti2 import VKITTI2
import numpy as np

def get_monodepth_vkitti2(split):
    return VKITTI2(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/nerf_public/VKITTI2",
        split=split
    )

def get_monodepth_hypersim(split):
    return HyperSim(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads"
    )

class MixedDataset(Dataset):
    def __init__(self, dataset_list, mix_rate_list, preprocess_list=None):
        self.datasets = list(dataset_list)
        self.datasets_len = np.array([len(ds) for ds in self.datasets])
        self.datasets_preprocess = preprocess_list if preprocess_list is not None else [None] * len(self.datasets)
        self.mix_rates = np.array(mix_rate_list)
        self.mix_rates = self.mix_rates / self.mix_rates.sum()
        self.mix_rates_acc = np.cumsum(self.mix_rates)
        assert all(map(lambda ds: isinstance(ds, Dataset), self.datasets)), \
            "MixedDataset: not all datasets are Dataset class"

    def __getitem__(self, index):
        di = np.searchsorted(self.mix_rates_acc, np.random.rand())
        ds = self.datasets[di]
        df = self.datasets_preprocess[di]
        sample = ds[index % self.datasets_len[di]]
        if callable(df):
            sample = df(sample)
        return sample

    def __len__(self):
        return self.datasets_len.max()

