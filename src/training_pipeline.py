from src.config_loaders.training_config_loader import TrainingConfig
from src.base_pipeline import BasePipeline
from colorama import Fore, Style
from datasets import load_from_disk
from src.modeling.model_builder import ModelBuilder
from torchvision.transforms import Compose
from src.utils.schema import BatchSchema


class TrainingPipeline(BasePipeline):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def run(self) -> None:

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")

        # Load train and validation sets
        print(f"{Fore.YELLOW}Loading data from specified paths...{Style.RESET_ALL}")
        input_dir = self.config.input_dir
        try:
            train_dataset = load_from_disk(f"{input_dir}/train")
            val_dataset = load_from_disk(f"{input_dir}/val")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Fore.RED}Could not find train or val datasets at {input_dir}{Style.RESET_ALL}"
            )

        # Create label to class ID mapping
        train_labels = train_dataset.features["labels"].names
        val_labels = val_dataset.features["labels"].names
        assert set(train_labels) == set(
            val_labels
        ), f"{Fore.RED}Train and validation sets have different labels{Style.RESET_ALL}"
        label2id, id2label = dict(), dict()
        for i, label in enumerate(train_labels):
            label2id[label] = i
            id2label[i] = label

        # Build the model
        print(f"{Fore.YELLOW}Creating Model...{Style.RESET_ALL}")
        model_builder = ModelBuilder(
            model_name=self.config.model_name,
            num_labels=len(train_labels),
            id2label=id2label,
            label2id=label2id,
            enable_gpu=self.config.enable_gpu,
        )
        _transforms, model, device = model_builder.initialize()

        # Image preprocessing
        def apply_transforms(batch: dict) -> dict:
            batch[BatchSchema.IMAGE_PROCESSED] = [
                _transforms(img.convert("RGB")) for img in batch["image"]
            ]
            del batch["image"]
            return batch

        train_dataset = train_dataset.with_transform(apply_transforms)
        val_dataset = val_dataset.with_transform(apply_transforms)

        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")
