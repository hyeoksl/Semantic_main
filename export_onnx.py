"""Export an experiment's best checkpoint to an ONNX model."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import yaml

import model as models


class ExportModel(nn.Module):
    """Express the complex AWGN channel using ONNX-compatible real tensors."""

    def __init__(self, model, snr, spp):
        super().__init__()
        self.model = model
        channels = model.decoder[0].in_channels
        self.active_channels = max(2, min(channels, 2 * round(spp / model.spp * (channels // 2))))
        self.noise_scale = 10 ** (snr / 10)

    def forward(self, image):
        encoded = self.model.encoder(image)
        active = encoded[:, :self.active_channels]
        # Complex noise has variance 1/2 in each real component. The original
        # complex signal power is twice the mean squared real-component value.
        noise_std = torch.sqrt(torch.mean(active ** 2) / self.noise_scale)
        received = active + noise_std * torch.randn_like(active)
        padded = torch.cat((received, torch.zeros_like(encoded[:, self.active_channels:])), dim=1)
        reconstructed = self.model.decoder(padded).clamp(0, 1)
        return reconstructed[:, :, :image.shape[2], :image.shape[3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', type=Path, help='Experiment directory containing config.yaml and best_ckpt_ep*.pth.tar')
    parser.add_argument('--snr', type=float, help='Export SNR in dB (default: midpoint of training range)')
    parser.add_argument('--spp', type=float, help='Export spp (default: configured maximum)')
    args = parser.parse_args()

    experiment = args.experiment_dir.expanduser()
    config_path = experiment / 'config.yaml'
    if not config_path.is_file():
        parser.error(f'Config not found: {config_path}')
    checkpoints = list(experiment.glob('best_ckpt_ep*.pth.tar'))
    if len(checkpoints) != 1:
        parser.error(f'Expected exactly one best checkpoint in {experiment}, found {len(checkpoints)}')

    # BaseLoader reads both ordinary YAML and legacy jsonargparse object tags
    # without instantiating arbitrary Python objects.
    with config_path.open() as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)['model']
    model_class = getattr(models, config['type'], None)
    if model_class is None or not issubclass(model_class, models.DeepJSCC_lite):
        parser.error(f"Unsupported model type: {config['type']}")
    max_spp = float(config['spp'])
    spp = args.spp if args.spp is not None else max_spp
    snr = args.snr if args.snr is not None else (float(config['min_SNRdB']) + float(config['max_SNRdB'])) / 2
    if not math.isfinite(snr) or not math.isfinite(spp) or not 0 < spp <= max_spp:
        parser.error('SNR must be finite and spp must be positive and at most the configured maximum')

    model = model_class(spp=max_spp, dim=int(config['dim']))
    checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    model.load_state_dict({key.removeprefix('module.'): value for key, value in state.items()})
    model.eval()
    export_model = ExportModel(model, snr, spp).eval()

    try:
        name = experiment.resolve().relative_to(Path('logs').resolve())
    except ValueError:
        name = Path(experiment.resolve().name)
    output = name.parent / f'{name.name}.onnx'
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model, (torch.zeros(1, 3, 256, 256),), str(output),
        input_names=['input_image'], output_names=['reconstructed_image'],
        dynamic_axes={
            'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
            'reconstructed_image': {0: 'batch_size', 2: 'height', 3: 'width'},
        },
        opset_version=17, dynamo=False,
    )
    print(f'Exported {checkpoints[0]} to {output} (SNR={snr:g} dB, spp={spp:g})')


if __name__ == '__main__':
    main()
