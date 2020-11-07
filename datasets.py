import os
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
from functional import seq
from PIL import Image
from pytorch_lightning.core import datamodule
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import Compose


class VideoDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.texture_images = os.listdir(os.path.join(f"{image_dir}"))
        self.transform = transforms.Compose(
            [transforms.Resize(128), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.texture_images)

    def __getitem__(self, index):
        texture_image = self.texture_images[index]
        image_path = os.path.join(f"{self.image_dir}", texture_image)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image


class DeepFashionDataset(Dataset):
    def __init__(
        self,
        file_path: str,
    ) -> None:
        self.file_paths = (
            seq(Path(file_path).read_text().split("\n"))
            .filter(lambda x: len(x) > 0)
            .sorted()
            .list()
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Any:
        sample_index = index
        with h5py.File(self.file_paths[index], mode="r") as h5_file:
            frame = h5_file["frame"][:]
            texture = h5_file["texture"][:]
            uv = (h5_file["uv"][:] * 255).astype(np.uint8)
            instances = h5_file["i"][:]

        frame = torch.from_numpy(frame).float().div(255).permute((2, 0, 1))
        texture = torch.from_numpy(texture).float().div(255).permute((2, 0, 1))
        iuv = torch.from_numpy(
            np.concatenate((instances[..., None], uv), axis=-1)
        ).permute((2, 0, 1))

        return {
            "sample_index": sample_index,
            "frame": frame,
            "texture": texture,
            "iuv": iuv,
        }


class DeepFashionDataModule(datamodule.LightningDataModule):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        batch_size: int,
        num_workers: int,
        iuv_transforms: Optional[Compose] = None,
        texture_transforms: Optional[Compose] = None,
    ) -> None:
        super().__init__()

        self.train_path = train_path
        self.valid_path = valid_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.iuv_transforms = iuv_transforms
        self.texture_transforms = texture_transforms

    def prepare_data(self, *args, **kwargs) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = DeepFashionDataset(self.train_path)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        dataset = DeepFashionDataset(self.valid_path)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self, *args, **kwargs):
        pass

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device
    ) -> Any:
        return {key: elem.to(device) for key, elem in batch.items()}
