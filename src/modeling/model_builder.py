from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.nn import Module
from colorama import Fore, Style
from typing import Optional
import torch


class ModelBuilder:

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        enable_gpu: Optional[bool] = False,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.enable_gpu = enable_gpu

    def build_transforms(self) -> Compose:
        """Build image preprocessing transforms"""
        image_processor = AutoImageProcessor.from_pretrained(
            self.model_name, use_fast=True
        )
        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )

        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )

        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
        return _transforms

    # def transforms(self, batch: dict) -> dict:
    #     batch['image_processed'] = [self._transforms(img.convert('RGB')) for img in batch['image']]
    #     del batch['image'] # delete original image for memory optimization
    #     return batch

    def build_model(self) -> Module:
        """Load pretrained model"""
        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        print(f"{Fore.CYAN}Model loaded.{Style.RESET_ALL}")

        return model

    def select_device(self) -> torch.device:
        "Select device for training"
        if self.enable_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        return device

    def initialize(self):
        _transforms = self.build_transforms()
        model = self.build_model()
        device = self.select_device()
        return _transforms, model, device
