# Stable Diffusion as 3D Reconstruction Foundation ModelM (SD-RFM)

We now focusing on finetuning SD (similar to Marigold) to achieve multi-view stereo.

## Structure
`datasets/`: dataloaders

`*_pipeline.py`: the pipeline definition

`*_inference.py`: the inference code for specific pipeline

`*_train.py`: the training code for specific pipeline

### Notes
- The `sd_train.py`, `sd_pipeline`, `train.sh` are from official examples, we can build our trainer based on the examples.

## Useful Variable:

```
conda activate /cpfs01/shared/pjlab-lingjun-landmarks/mali1/miniconda3/envs/diffuser
git config --global http.proxy http://58.34.83.134:31280/
http_proxy=http://58.34.83.134:31280/
https_proxy=http://58.34.83.134:31280/
HTTP_PROXY=http://58.34.83.134:31280/
HTTPS_PROXY=http://58.34.83.134:31280/
HF_HOME=/cpfs01/shared/pjlab-lingjun-landmarks/mali1/.cache
```