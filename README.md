# sem-base

Train the DeepJSCC image transmission model with Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python train.py --config_file configs/default.yaml
```

The project uses PyTorch 2.9 and CUDA 12.8 wheels on Linux and Windows. A compatible NVIDIA driver is needed for GPU training; the visible GPU with the most free memory is selected. Use `--use_cpu` for CPU training, or set `CUDA_VISIBLE_DEVICES` to limit GPU selection. See the full options with `uv run python train.py --config_file configs/default.yaml --help`.

Set `dataset.train_dataset` and `dataset.valid_dataset` in the configuration to image folders, such as `~/datasets/ImageNet8000` and `~/datasets/Kodak`. Images are discovered recursively, so filenames and image counts need not follow a fixed pattern. Supported extensions are JPEG, PNG, BMP, WebP, and TIFF. Both training and validation use random 256×256 crops after resizing; training also adds a small amount of noise. Each image is used once per epoch. Training output and checkpoints are written under `logs/BASELINE` by default. Configuration values can be overridden on the command line, for example `--dataset.train_dataset ~/datasets/ImageNet8000 --dataset.num_workers 0`.

The available models are `DeepJSCC_lite`, `DeepJSCC_lite_MLP`, and `DeepJSCC_lite_SwiGLU`, all with MSE loss. The latter two replace the decoder's first projection with a pointwise MLP or SwiGLU, respectively; select one with `model.type`. JPEG and other supported images are decoded as RGB with Pillow. The project does not require CompressAI, pynvml_utils, pytorch_model_summary, setproctitle, or turbojpeg.

For each training batch, SNR in dB is sampled uniformly between `model.min_SNRdB` and `model.max_SNRdB`, and spp is sampled uniformly between `model.min_spp` and `model.spp`. The sampled spp is rounded to the nearest available complex-symbol channel count. Validation uses the midpoint SNR and maximum spp.
