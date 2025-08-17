from torchvision.transforms import Compose, RandomResizedCrop, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from colorama import Fore, Style
from typing import Optional
import torch
from torch import nn


class ModelBuilder:

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        enable_gpu: Optional[bool] = False,
        nb_layers_to_freeze: Optional[int] = None,
        dropout_rate: Optional[float] = None,
    ):
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.enable_gpu = enable_gpu
        self.nb_layers_to_freeze = nb_layers_to_freeze
        self.dropout_rate = dropout_rate

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

    def _print_trainable_parameters(self, model: nn.Module) -> None:
        print(f"{Fore.CYAN}Trainable parameters:{Style.RESET_ALL}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f" - {name}")

    def build_model(self) -> nn.Module:
        """Load pretrained model"""

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        # Control Dropout rate if specified
        if self.dropout_rate is not None:
            config.hidden_dropout_prob = self.dropout_rate
            config.attention_probs_dropout_prob = self.dropout_rate
            print(
                f"{Fore.CYAN}Dropout rate set to {self.dropout_rate}.{Style.RESET_ALL}"
            )
        else:
            print(f"{Fore.CYAN}No dropout rate specified.{Style.RESET_ALL}")

        model = AutoModelForImageClassification.from_pretrained(
            self.model_name,
            config=config,
        )
        print(f"{Fore.CYAN}Model loaded.{Style.RESET_ALL}")

        # Freeze first layers if specified
        if self.nb_layers_to_freeze is not None:
            print(
                f"{Fore.YELLOW}Freezing first {self.nb_layers_to_freeze} encoder layers...{Style.RESET_ALL}"
            )
            for name, param in model.vit.named_parameters():
                for layer_idx in range(self.nb_layers_to_freeze):
                    if name.startswith(f"encoder.layer.{layer_idx}."):
                        param.requires_grad = False

        # Print trainable parameters
        self._print_trainable_parameters(model)

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
