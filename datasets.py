import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VideoDataset(Dataset):
    def __init__(self, image_dir):
        super().__init__()
        self.image_dir = image_dir
        self.texture_images = os.listdir(os.path.join(f"{image_dir}"))
        self.transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                ),
            ]
        )

    def __len__(self):
        return len(self.texture_images)

    def __getitem__(self, index):
        texture_image = self.texture_images[index]
        image_path = os.path.join(f"{self.image_dir}", texture_image)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image
