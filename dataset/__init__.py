from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from dataset.diode import DIODE
from dataset.ibims import iBims
from dataset.kitti import KITTI

from dataset.nyu_v2 import Nyu_v2
from .hypersim import HyperSim
from .vkitti2 import VKITTI2
from .preprocesses import set_depth_normalize_fn, preprocess_functions
import numpy as np


def get_monodepth_vkitti2(split):
    return VKITTI2(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/nerf_public/VKITTI2",
        preprocess=preprocess_functions["vkitti"][split],
        split=split,
    )


def get_monodepth_hypersim(split):
    return HyperSim(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads",
        preprocess=preprocess_functions["hypersim"][split],
        split=split
    )
    
def get_monodepth_nyu(config, split):
    return Nyu_v2(
        config=config,
        split=split
    )

def get_monodepth_kitti(config, split):
    return KITTI(
        config=config,
        split=split
    )

def get_monodepth_ibims(config, split):
    return iBims(
        config=config,
        split=split
    )

def get_monodepth_diode(data_dir_root, split):
    return DIODE(
        data_dir_root = data_dir_root,
        preprocess=preprocess_functions["diode"][split],
        split=split
    )

# mixture in dataset level, allow different datasets for each batch
class MixedDataset(Dataset):
    def __init__(self, dataset_list, mix_rate_list):
        self.datasets = list(dataset_list)
        self.datasets_len = np.array([len(ds) for ds in self.datasets])
        self.mix_rates = np.array(mix_rate_list)
        self.mix_rates = self.mix_rates / self.mix_rates.sum()
        self.mix_rates_acc = np.cumsum(self.mix_rates)
        assert all(map(lambda ds: isinstance(ds, Dataset), self.datasets)), \
            "MixedDataset: not all datasets are Dataset class"

    def __getitem__(self, index):
        di = np.searchsorted(self.mix_rates_acc, np.random.rand())
        ds = self.datasets[di]
        sample = ds[index % self.datasets_len[di]]
        return sample

    def __len__(self):
        return self.datasets_len.max()


# mixture in sampler level, force same dataset for each batch
class MixedBatchSampler(BatchSampler):
    def __init__(self, batch_size, datasets, mix_rates):
        self.datasets = datasets
        self.batch_size = batch_size
        mix_rates = np.array(mix_rates)
        self.mix_rates = mix_rates / mix_rates.sum()
        self.mix_rates_acc = np.cumsum(self.mix_rates)

        self.dataset_sizes = np.array([len(dataset) for dataset in self.datasets])
        # Create separate samplers for each dataset
        self.samplers = [RandomSampler(dataset) for dataset in datasets]

    def __iter__(self):
        dataset_iters = [iter(sampler) for sampler in self.samplers]

        for _ in range(len(self)):
            di = np.searchsorted(self.mix_rates_acc, np.random.rand())
            it = dataset_iters[di]
            batch = [next(it) + self.dataset_sizes[:di].sum() for _ in range(self.batch_size)]
            yield batch

    def __len__(self):
        return max(self.dataset_sizes) // self.batch_size


class DepthEvalDataLoader(object):
    def __init__(self, config, **kwargs):
        self.config = config
        if config.dataset == 'nyu':
            self.testing_samples = get_monodepth_nyu(config, 'test')
        
        if config.dataset == 'kitti':
            self.testing_samples = get_monodepth_kitti(config, 'test')

        if config.dataset == 'ibims':
            self.testing_samples = get_monodepth_ibims(config, 'test')

        if config.dataset == 'diml_indoor':
            self.data = get_diml_indoor_loader(
                data_dir_root=config.diml_indoor_root, batch_size=1, num_workers=1)
            return

        if config.dataset == 'diml_outdoor':
            self.data = get_diml_outdoor_loader(
                data_dir_root=config.diml_outdoor_root, batch_size=1, num_workers=1)
            return

        if "diode" in config.dataset:
            self.testing_samples = get_monodepth_diode(config[config.dataset+"_root"], 'test')

        if config.dataset == 'hypersim_test':
            self.testing_samples = get_monodepth_hypersim('test')

        if config.dataset == 'vkitti2_test':
            self.testing_samples = get_monodepth_vkitti2('test')

        if config.dataset == 'ddad':
            self.data = get_ddad_loader(config.ddad_root, resize_shape=(
                352, 1216), batch_size=1, num_workers=1)
            return
        
        self.data = DataLoader(self.testing_samples, 1,
            shuffle=kwargs.get("shuffle_test", False),
            num_workers=1,
            pin_memory=False,
            sampler=None)
        