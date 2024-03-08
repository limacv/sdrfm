from torch.utils.data import Dataset
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from .hypersim import HyperSimMono
from .preprocesses import set_normal_normalize_fn, set_depth_normalize_fn, set_hypersim_resolution
import numpy as np


def get_mono_hypersim(split, assets):
    from .preprocesses import preprocess_functions
    return HyperSimMono(
        data_dir_root="/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads",
        assets=assets,
        preprocess=preprocess_functions["hypersim"][split],
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

