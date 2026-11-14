"""Run separate encoder and decoder ONNX models with a simulated AWGN channel."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('encoder', type=Path)
    parser.add_argument('decoder', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--snr', type=float, default=10.0, help='Simulated channel SNR in dB')
    parser.add_argument('--output', type=Path, default=Path('reconstruction.png'))
    args = parser.parse_args()
    with Image.open(args.image.expanduser()) as image:
        input_image = np.asarray(image.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    height, width = input_image.shape[-2:]
    encoder = ort.InferenceSession(str(args.encoder.expanduser()), providers=['CPUExecutionProvider'])
    decoder = ort.InferenceSession(str(args.decoder.expanduser()), providers=['CPUExecutionProvider'])
    symbols = encoder.run(['symbols'], {'input_image': input_image})[0]
    noise_std = np.sqrt(np.mean(symbols ** 2) / 10 ** (args.snr / 10))
    received = symbols + np.random.standard_normal(symbols.shape).astype(np.float32) * noise_std
    result = decoder.run(['reconstructed_image'], {
        'received': received, 'height': np.array(height, dtype=np.int64), 'width': np.array(width, dtype=np.int64),
    })[0]
    output = (np.clip(result[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)
    print(f'Saved reconstruction to {args.output}')


if __name__ == '__main__':
    main()
