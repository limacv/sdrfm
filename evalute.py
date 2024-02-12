import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
import pdb
from pprint import pprint
from dataset import DepthEvalDataLoader
from tqdm import tqdm
from PIL import Image

from marigold_pipeline import MarigoldPipeline
from utils.config import ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR, get_config
from utils.misc import *
from utils.general_utils import *

def eval_marigold(args, pipeline, round_vals=True, round_precision=3):
    config = get_config("eval", args.dataset)
    pprint(config)
    print(f"Evaluating Marigold on {args.dataset}...")
    test_loader = DepthEvalDataLoader(config).data
    
    metrics = RunningAverageDict()
    save_dict={}
    
    idx_thre = 5
    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        # pdb.set_trace()
        image, depth = sample['image'].numpy(), sample['depth'].numpy()
        image = image.squeeze()
        depth = depth.squeeze()
        if image.shape[0] == 3:
            image = np.transpose(image,(1,2,0))
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        
        with torch.autocast("cuda"):
            res = pipeline(image, denoising_steps=10, ensemble_size=10)
            pred = res.depth_np
        
        # mask first
        valid_mask = compute_masks(depth, pred, config=config)
        pred_metric = compute_scale_pred(depth, pred, valid_mask, config=config)
        
        metric = compute_errors(depth[valid_mask], pred_metric[valid_mask])
        metrics.update(metric)
        
        # Save image, depth, pred for visualization
        gt_depth_color = depth2color(depth, pred_metric.min(), pred_metric.max(), valid_mask=valid_mask)
        pred_depth_color = depth2color(pred_metric, pred_metric.min(), pred_metric.max())

        save_dict.update({
                f"{i:04d}_images": image,
                f"{i:04d}_gt": gt_depth_color,
                f"{i:04d}_pred": pred_depth_color
            })
    
        if i>idx_thre:
            break
        
    # save image
    save_path = os.path.join("outputs", f"{dataset}")
    os.makedirs(save_path,exist_ok=True)
    
    for basename, img in save_dict.items():
        img.save(os.path.join(save_path, basename + ".png"))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
        
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    
    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")

if __name__ == "__main__":
    
    
    # choose which dataset for eval
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")
    parser.add_argument("-m", "--model", type=str, required=False,
                        default='outputs/marigold/fix_bug', help="which model to evaluate on")
    args, unknown_args = parser.parse_known_args()
    
    # load the pretrained pipeline
    pipe = MarigoldPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,                # (optional) Run with half-precision (16-bit float).
    )
    pipe.to("cuda")
    
    if "ALL_INDOOR" in args.dataset:
        datasets = ALL_INDOOR
    elif "ALL_OUTDOOR" in args.dataset:
        datasets = ALL_OUTDOOR
    elif "ALL" in args.dataset:
        datasets = ALL_EVAL_DATASETS
    elif "," in args.dataset:
        datasets = args.dataset.split(",")
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        eval_marigold(args, pipe)
    