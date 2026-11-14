import io
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

__all__ = ['load_dataloaders']


class AddContinuousNoise:
    def __call__(self, image):
        noise = (torch.rand(image.shape[-2:], device=image.device) - 0.5) / 256
        return (image + noise).clamp_(0, 1)


class FolderImageDataset(Dataset):
    """Load images from a directory, including any nested directories."""

    def __init__(self, root_dir, transform):
        root = Path(root_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {root}")
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        paths = sorted(path for path in root.rglob('*')
                       if path.is_file() and path.suffix.lower() in extensions)
        if not paths:
            raise ValueError(f"No supported images found in dataset directory: {root}")
        self.image_bytes = [path.read_bytes() for path in paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_bytes)

    def __getitem__(self, idx):
        with Image.open(io.BytesIO(self.image_bytes[idx])) as decoded:
            image = decoded.convert('RGB')
        return self.transform(image)


def load_dataloaders(train_dataset, valid_dataset, train_batch_size, valid_batch_size, num_workers):
    transforms = [
        v2.RandomResize(640, 1200),
        v2.RandomCrop((256, 256)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    train_ds = FolderImageDataset(train_dataset, v2.Compose([*transforms, AddContinuousNoise()]))
    valid_ds = FolderImageDataset(valid_dataset, v2.Compose(transforms))

    train_dl = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=3 if num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )
    return train_dl, valid_dl
