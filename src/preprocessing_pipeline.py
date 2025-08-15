from src.config_loaders.preprocessing_config_loader import PreprocessingConfig
from src.base_pipeline import BasePipeline
from colorama import Fore, Style
from datasets import load_dataset
from src.utils.utils_toolbox import describe_dataset


class PreprocessingPipeline(BasePipeline):
    def __init__(self, config: PreprocessingConfig):
        super().__init__(config)

    def run(self) -> None:

        print(f"{Fore.GREEN}Starting preprocessing pipeline...{Style.RESET_ALL}")

        # Load dataset
        dataset = load_dataset(self.config.huggingface_dataset_name, split="train")
        if len(dataset) > 0:
            print(f"{Fore.GREEN}Dataset loaded successfully{Style.RESET_ALL}")
        else:
            raise ValueError(
                f"{Fore.RED}Dataset is empty! Check the dataset name ({self.config.huggingface_dataset_name}) the config file.{Style.RESET_ALL}"
            )

        # Split dataset to train, validation and test sets
        train_test_splits = dataset.train_test_split(
            test_size=self.config.test_size, seed=42, shuffle=True
        )
        train_val_splits = train_test_splits["train"].train_test_split(
            test_size=self.config.validation_size, seed=42, shuffle=True
        )

        train_dataset = train_val_splits["train"]
        train_labels = set(train_dataset.features["labels"].names)
        describe_dataset(dataset=train_dataset, split="train")

        validation_dataset = train_val_splits["test"]
        val_labels = set(validation_dataset.features["labels"].names)
        describe_dataset(dataset=validation_dataset, split="validation")

        test_dataset = train_test_splits["test"]
        test_labels = set(test_dataset.features["labels"].names)
        describe_dataset(dataset=test_dataset, split="test")

        # Ensure that sets have the same labels
        assert (
            train_labels == val_labels
        ), f"{Fore.RED}Train and validation sets have different labels{Style.RESET_ALL}"
        assert (
            train_labels == test_labels
        ), f"{Fore.RED}Train and test sets have different labels{Style.RESET_ALL}"

        # Save datasets
        output_dir = self.config.output_dir
        train_dataset.save_to_disk(f"{output_dir}/train")
        validation_dataset.save_to_disk(f"{output_dir}/val")
        test_dataset.save_to_disk(f"{output_dir}/test")

        print(
            f"{Fore.GREEN}Preprocessing pipeline completed successfully!{Style.RESET_ALL}"
        )
