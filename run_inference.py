"""Run an exported ONNX model on an image and save the reconstruction."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path, help='Exported ONNX model')
    parser.add_argument('image', type=Path, help='Input image')
    parser.add_argument('--output', type=Path, default=Path('reconstruction.png'))
    args = parser.parse_args()

    with Image.open(args.image.expanduser()) as image:
        image = image.convert('RGB')
        input_image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    session = ort.InferenceSession(str(args.model.expanduser()), providers=['CPUExecutionProvider'])
    result = session.run(['reconstructed_image'], {'input_image': input_image})[0]
    output = (np.clip(result[0].transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output)
    print(f'Saved reconstruction to {args.output}')


if __name__ == '__main__':
    main()
