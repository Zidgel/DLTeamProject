from PIL import Image
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets import load_dataset
from IPython.display import display
from trainer import trainer
import torchvision.models as models
from trainer import trainer
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF


class fst_api:
    def __init__(self, style_image: Image, dataset_path: str, epochs, lr, style_beta):
        self.style_transformation = self._get_style_transform(style_image)
        self.content_transformation = self._get_content_transform(style_image)
        self.dataloader = self._init_dataloader(dataset_path)
        self.style_image = style_image
        self.epochs = epochs
        self.lr = lr
        self.style_beta = style_beta
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA!")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS!")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")
        self.tr = None

    def _get_style_transform(self, image: Image):
        width, height = image.size
        width = (int)(width / 4) * 4
        height = (int)(height / 4) * 4
        style_transform = transforms.Compose(
            [
                transforms.Resize(size=(height, width)),
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
            ]
        )
        return style_transform

    def _get_content_transform(self, image: Image):
        width, height = image.size
        width = (int)(width / 4) * 4
        height = (int)(height / 4) * 4
        content_transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert to tensor
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
            ]
        )
        return content_transform

    def _init_dataloader(self, path: str):
        # Load dataset using ImageFolder
        dataset = datasets.ImageFolder(root=path, transform=self.style_transformations)

        # Create DataLoader
        return DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    def train_model(self):
        tensor = self.style_transformation(self.style_image)
        self.tr = trainer(
            tensor, self.epochs, self.dataloader, self.device, self.lr, self.style_beta
        )
        self.tr.train()

    def style_transfer(self, content_image: Image):
        content_tensor = (
            self.content_transformation(content_image).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            output = self.tr.model.net.forward(content_tensor)
            output = (output.clamp(-1, 1) + 1) / 2
            stylized_image = TF.to_pil_image(output.squeeze(0).cpu())
            stylized_image.show()
