# sem-base

Train the DeepJSCC image transmission model with Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python train.py --config_file configs/default.yaml
```

The project uses PyTorch 2.9 and CUDA 12.8 wheels on Linux and Windows. A compatible NVIDIA driver is needed for GPU training; the visible GPU with the most free memory is selected. Use `--use_cpu` for CPU training, or set `CUDA_VISIBLE_DEVICES` to limit GPU selection. See the full options with `uv run python train.py --config_file configs/default.yaml --help`.

Set `dataset.train_dataset` and `dataset.valid_dataset` in the configuration to image folders, such as `~/datasets/ImageNet8000` and `~/datasets/Kodak`. Images are discovered recursively, so filenames and image counts need not follow a fixed pattern. Supported extensions are JPEG, PNG, BMP, WebP, and TIFF. Both training and validation use random 256×256 crops after resizing; training also adds a small amount of noise. Each image is used once per epoch. Training output and checkpoints are written under `logs/BASELINE` by default. Configuration values can be overridden on the command line, for example `--dataset.train_dataset ~/datasets/ImageNet8000 --dataset.num_workers 0`.

The `DeepJSCC_lite` configuration uses separate `DeepJSCCEncoder` and `DeepJSCCDecoder` modules. The encoder takes an RGB tensor and an operating spp and returns real-valued symbol channels in `(batch, channels, symbol_height, symbol_width)` order. Each adjacent pair of channels represents the real and imaginary parts of a complex symbol. The spatial dimensions are the image dimensions divided by 16 and rounded up. The decoder takes the received symbol tensor and the original height and width, restores any unused channels internally, and returns the reconstructed image at its original dimensions. Both modules use the same maximum spp and dimension settings. Training simulates an AWGN channel between them; the channel is not part of either module.

For each training batch, SNR in dB is sampled uniformly between `model.min_SNRdB` and `model.max_SNRdB`, and spp is sampled uniformly between `model.min_spp` and `model.spp`. The sampled spp is rounded to the nearest available complex-symbol channel count. Validation uses the midpoint SNR and maximum spp.

Export the best checkpoint from an experiment directory and run an image through both ONNX models with a simulated channel:

```sh
uv run export_onnx.py logs/BASELINE
uv run run_inference.py BASELINE_encoder.onnx BASELINE_decoder.onnx input.png --snr 10 --output reconstruction.png
```

The exporter writes separate `<experiment>_encoder.onnx` and `<experiment>_decoder.onnx` files and preserves paths nested below `logs/`. The encoder takes float32 `input_image` of shape `(batch, 3, height, width)`, scaled to [0, 1], and outputs float32 `symbols`. The decoder takes float32 `received` with the same layout, plus scalar int64 `height` and `width`, and outputs `reconstructed_image`. Send the original image height and width as side information with the symbols. Image and symbol spatial dimensions and batch size are dynamic; the decoder also accepts different valid even channel counts up to its configured maximum.

The ONNX encoder's spp is fixed at export time. It defaults to the configured maximum and can be changed with `--spp`; the decoder works with any supported channel count. The example inference script uses simulated AWGN, controlled by `--snr`, between the two ONNX sessions. A real transmitter and receiver should use their physical channel instead. Existing combined ONNX files must be re-exported as separate models.
