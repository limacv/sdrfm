import json
import os

from utils.easydict import EasyDict as edict

from utils.arg_utils import infer_type
import pathlib
import platform


ROOT = pathlib.Path(__file__).parent.parent.resolve()

HOME_DIR = '/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/depth_data'

COMMON_CONFIG = {
    "save_dir": "outputs",
}

DATASETS_CONFIG = {
    "kitti": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": os.path.join(HOME_DIR, "../KITTI/"),
        "gt_path": os.path.join(HOME_DIR, "../KITTI/gt_depth"),
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,  # 704
        "data_path_eval": os.path.join(HOME_DIR, "../KITTI/"),
        "gt_path_eval": os.path.join(HOME_DIR, "../KITTI/gt_depth"),
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",

        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "kitti_test": {
        "dataset": "kitti",
        "min_depth": 0.001,
        "max_depth": 80,
        "data_path": os.path.join(HOME_DIR, "../KITTI/"),
        "gt_path": os.path.join(HOME_DIR, "../KITTI/gt_depth"),
        "filenames_file": "./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "input_height": 352,
        "input_width": 1216,
        "data_path_eval": os.path.join(HOME_DIR, "../KITTI/"),
        "gt_path_eval": os.path.join(HOME_DIR, "../KITTI/gt_depth"),
        "filenames_file_eval": "./train_test_inputs/kitti_eigen_test_files_with_gt.txt",

        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,

        "do_random_rotate": False,
        "degree": 1.0,
        "do_kb_crop": True,
        "garg_crop": True,
        "eigen_crop": False,
        "use_right": False
    },
    "nyu": {
        "dataset": "nyu",
        "avoid_boundary": False,
        "min_depth": 1e-3,   # originally 0.1
        "max_depth": 10,
        "data_path": os.path.join(HOME_DIR, "nyu_depth_v2/sync/"),
        "gt_path": os.path.join(HOME_DIR, "nyu_depth_v2/sync/"),
        "filenames_file": "./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        "input_height": 480,
        "input_width": 640,
        "data_path_eval": os.path.join(HOME_DIR, "nyu_depth_v2/official_splits/test/"),
        "gt_path_eval": os.path.join(HOME_DIR, "nyu_depth_v2/official_splits/test/"),
        "filenames_file_eval": "./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "min_depth_diff": -10,
        "max_depth_diff": 10,

        "do_random_rotate": True,
        "degree": 1.0,
        "do_kb_crop": False,
        "garg_crop": False,
        "eigen_crop": True
    },
    "ibims": {
        "dataset": "ibims",
        "ibims_root": os.path.join(HOME_DIR, "ibims/ibims1_core_raw/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "sunrgbd": {
        "dataset": "sunrgbd",
        "sunrgbd_root": os.path.join(HOME_DIR, "SUNRGBD/test/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 8,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diml_indoor": {
        "dataset": "diml_indoor",
        "diml_indoor_root": os.path.join(HOME_DIR, "shortcuts/datasets/diml_indoor_test/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 0,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diml_outdoor": {
        "dataset": "diml_outdoor",
        "diml_outdoor_root": os.path.join(HOME_DIR, "diml/outdoor/test/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": False,
        "min_depth_eval": 2,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80
    },
    "diode_indoor": {
        "dataset": "diode_indoor",
        "diode_indoor_root": os.path.join(HOME_DIR, "../DIODE/val/indoor/"),
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 10,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "diode_outdoor": {
        "dataset": "diode_outdoor",
        "diode_outdoor_root": os.path.join(HOME_DIR, "../DIODE/val/outdoor/"),
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80
    },
    "hypersim_test": {
        "dataset": "hypersim_test",
        "data_dir_root": "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/jianglihan/Hypersim/portable_hard_drive/downloads",
        "eigen_crop": True,
        "garg_crop": False,
        "do_kb_crop": False,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 10
    },
    "vkitti2_test": {
        "dataset": "vkitti2_test",
        "data_dir_root": "/cpfs01/shared/pjlab-lingjun-landmarks/pjlab-lingjun-landmarks_hdd/nerf_public/VKITTI2",
        "eigen_crop": False,
        "garg_crop": True,
        "do_kb_crop": True,
        "min_depth_eval": 1e-3,
        "max_depth_eval": 80,
        "min_depth": 1e-3,
        "max_depth": 80,
    }
}

ALL_INDOOR = ["nyu", "ibims", "diode_indoor", "hypersim_test"]
ALL_OUTDOOR = ["kitti", "diml_outdoor", "diode_outdoor",  "vkitti2"]
ALL_EVAL_DATASETS = ALL_INDOOR + ALL_OUTDOOR

COMMON_TRAINING_CONFIG = {
    "dataset": "mix",
    "distributed": True,
    "workers": 16,
    "clip_grad": 0.1,

    "aug": True,
    "random_crop": False,
    "random_translate": False,
    "translate_prob": 0.2,
    "max_translation": 100,
}


def flatten(config, except_keys=('bin_conf')):
    def recurse(inp):
        if isinstance(inp, dict):
            for key, value in inp.items():
                if key in except_keys:
                    yield (key, value)
                if isinstance(value, dict):
                    yield from recurse(value)
                else:
                    yield (key, value)

    return dict(list(recurse(config)))


def split_combined_args(kwargs):
    """Splits the arguments that are combined with '__' into multiple arguments.
       Combined arguments should have equal number of keys and values.
       Keys are separated by '__' and Values are separated with ';'.
       For example, '__n_bins__lr=256;0.001'

    Args:
        kwargs (dict): key-value pairs of arguments where key-value is optionally combined according to the above format. 

    Returns:
        dict: Parsed dict with the combined arguments split into individual key-value pairs.
    """
    new_kwargs = dict(kwargs)
    for key, value in kwargs.items():
        if key.startswith("__"):
            keys = key.split("__")[1:]
            values = value.split(";")
            assert len(keys) == len(
                values), f"Combined arguments should have equal number of keys and values. Keys are separated by '__' and Values are separated with ';'. For example, '__n_bins__lr=256;0.001. Given (keys,values) is ({keys}, {values})"
            for k, v in zip(keys, values):
                new_kwargs[k] = v
    return new_kwargs


def parse_list(config, key, dtype=int):
    """Parse a list of values for the key if the value is a string. The values are separated by a comma. 
    Modifies the config in place.
    """
    if key in config:
        if isinstance(config[key], str):
            config[key] = list(map(dtype, config[key].split(',')))
        assert isinstance(config[key], list) and all([isinstance(e, dtype) for e in config[key]]
                                                     ), f"{key} should be a list of values dtype {dtype}. Given {config[key]} of type {type(config[key])} with values of type {[type(e) for e in config[key]]}."



def check_choices(name, value, choices):
    # return  # No checks in dev branch
    if value not in choices:
        raise ValueError(f"{name} {value} not in supported choices {choices}")


KEYS_TYPE_BOOL = ["use_amp", "distributed", "use_shared_dict", "same_lr", "aug", "three_phase",
                  "prefetch", "cycle_momentum"]  # Casting is not necessary as their int casted values in config are 0 or 1


def get_config(mode='train', dataset=None, **overwrite_kwargs):
    """Main entry point to get the config for the model.

    Args:
        model_name (str): name of the desired model.
        mode (str, optional): "train" or "infer". Defaults to 'train'.
        dataset (str, optional): If specified, the corresponding dataset configuration is loaded as well. Defaults to None.
    
    Keyword Args: key-value pairs of arguments to overwrite the default config.

    The order of precedence for overwriting the config is (Higher precedence first):
        # 1. overwrite_kwargs
        # 2. "config_version": Config file version if specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{config_version}.json
        # 3. "version_name": Default Model version specific config specified in overwrite_kwargs. The corresponding config loaded is config_{model_name}_{version_name}.json
        # 4. common_config: Default config for all models specified in COMMON_CONFIG

    Returns:
        easydict: The config dictionary for the model.
    """

    check_choices("Mode", mode, ["train", "eval"])
    if mode == "train":
        check_choices("Dataset", dataset, ["hypersim", "vkitti", "mix", None])

    config = flatten({**COMMON_CONFIG, **COMMON_TRAINING_CONFIG})

    # update with overwrite_kwargs
    # Combined args are useful for hyperparameter search
    overwrite_kwargs = split_combined_args(overwrite_kwargs)
    config = {**config, **overwrite_kwargs}

    # Casting to bool   # TODO: Not necessary. Remove and test
    for key in KEYS_TYPE_BOOL:
        if key in config:
            config[key] = bool(config[key])

    # Model specific post processing of config
    parse_list(config, "n_attractors")

    if dataset is not None:
        config['dataset'] = dataset
        config = {**DATASETS_CONFIG[dataset], **config}
        

    typed_config = {k: infer_type(v) for k, v in config.items()}
    # add hostname to config
    config['hostname'] = platform.node()
    return edict(typed_config)

