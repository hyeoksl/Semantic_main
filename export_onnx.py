"""Export an experiment's encoder and decoder as separate ONNX models."""

import argparse
import math
from pathlib import Path

import torch
from torch import nn
import yaml

from model import DeepJSCCEncoder, DeepJSCCDecoder


class EncoderExport(nn.Module):
    def __init__(self, encoder, spp):
        super().__init__()
        self.encoder = encoder
        self.spp = spp

    def forward(self, image):
        return self.encoder(image, self.spp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_dir', type=Path)
    parser.add_argument('--spp', type=float, help='Encoder spp (default: configured maximum)')
    args = parser.parse_args()
    experiment = args.experiment_dir.expanduser()
    config_path = experiment / 'config.yaml'
    if not config_path.is_file():
        parser.error(f'Config not found: {config_path}')
    checkpoints = list(experiment.glob('best_ckpt_ep*.pth.tar'))
    if len(checkpoints) != 1:
        parser.error(f'Expected exactly one best checkpoint in {experiment}, found {len(checkpoints)}')
    with config_path.open() as file:
        config = yaml.load(file, Loader=yaml.BaseLoader)['model']
    if config['type'] != 'DeepJSCC_lite':
        parser.error(f"Unsupported model type: {config['type']}")
    max_spp = float(config['spp'])
    spp = max_spp if args.spp is None else args.spp
    if not math.isfinite(spp) or not 0 < spp <= max_spp:
        parser.error('spp must be finite, positive and at most the configured maximum')
    encoder = DeepJSCCEncoder(spp=max_spp, dim=int(config['dim'])).eval()
    decoder = DeepJSCCDecoder(spp=max_spp, dim=int(config['dim'])).eval()
    checkpoint = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    state = {key.removeprefix('module.'): value for key, value in checkpoint['state_dict'].items()}
    encoder.load_state_dict({k.removeprefix('encoder.'): v for k, v in state.items() if k.startswith('encoder.')})
    decoder.load_state_dict({k.removeprefix('decoder.'): v for k, v in state.items() if k.startswith('decoder.')})

    try:
        name = experiment.resolve().relative_to(Path('logs').resolve())
    except ValueError:
        name = Path(experiment.resolve().name)
    name.parent.mkdir(parents=True, exist_ok=True)
    encoder_path = name.parent / f'{name.name}_encoder.onnx'
    decoder_path = name.parent / f'{name.name}_decoder.onnx'
    image = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        symbols = encoder(image, spp)
    torch.onnx.export(
        EncoderExport(encoder, spp), (image,), str(encoder_path),
        input_names=['input_image'], output_names=['symbols'],
        dynamic_axes={'input_image': {0: 'batch', 2: 'height', 3: 'width'},
                      'symbols': {0: 'batch', 2: 'symbol_height', 3: 'symbol_width'}},
        opset_version=17, dynamo=False,
    )
    torch.onnx.export(
        decoder, (symbols, torch.tensor(256), torch.tensor(256)), str(decoder_path),
        input_names=['received', 'height', 'width'], output_names=['reconstructed_image'],
        dynamic_axes={'received': {0: 'batch', 1: 'channels', 2: 'symbol_height', 3: 'symbol_width'},
                      'reconstructed_image': {0: 'batch', 2: 'height', 3: 'width'}},
        opset_version=17, dynamo=False,
    )
    print(f'Exported {encoder_path} and {decoder_path} from {checkpoints[0]} (encoder spp={spp:g})')


if __name__ == '__main__':
    main()
