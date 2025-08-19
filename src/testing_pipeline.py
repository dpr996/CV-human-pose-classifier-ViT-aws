from src.config_loaders.testing_config_loader import TestingConfig
from colorama import Fore, Style
from src.base_pipeline import BasePipeline
from src.utils.schema import BatchSchema
from transformers import AutoModelForImageClassification
from datasets import load_from_disk
from src.modeling.model_builder import ModelBuilder


class TestingPipeline(BasePipeline):
    def __init__(self, config: TestingConfig):
        super().__init__(config)

    def run(self):

        print(f"{Fore.GREEN}Starting testing pipeline...{Style.RESET_ALL}")

        # Load test data
        print(f"{Fore.YELLOW}Loading test data from specified path...{Style.RESET_ALL}")
        input_dir = self.config.input_dir
        try:
            test_dataset = load_from_disk(f"{input_dir}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Fore.RED}Could not find test dataset at {input_dir}{Style.RESET_ALL}"
            )

        # Load the model
        print(
            f"{Fore.YELLOW}Loading trained model from {self.config.trained_model_path}{Style.RESET_ALL}"
        )

        labels = test_dataset.features["labels"].names
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}

        model_builder = ModelBuilder(
            model_name=self.config.model_name,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )
        _transforms, _, _ = model_builder.initialize()

        model = AutoModelForImageClassification.from_pretrained(
            self.config.trained_model_path,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        def apply_transforms(batch: dict) -> dict:
            batch[BatchSchema.PIXEL_VALUES] = [
                _transforms(img.convert("RGB")) for img in batch["image"]
            ]
            del batch["image"]
            return batch

        test_dataset = test_dataset.with_transform(apply_transforms)

        # Make predictions on test set
        print(f"{Fore.YELLOW}Making predictions on test set{Style.RESET_ALL}")
        preds, labels_list = [], []
        for sample in test_dataset:
            pixel_values = sample[BatchSchema.PIXEL_VALUES].unsqueeze(0)
            true_label = sample["labels"]

            outputs = model(pixel_values)
            pred_label_id = outputs.logits.argmax(-1).item()
            preds.append(pred_label_id)
            labels_list.append(true_label)

        print(f"{Fore.GREEN}Testing pipeline completed successfully!{Style.RESET_ALL}")
