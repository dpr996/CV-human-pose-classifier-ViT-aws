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

        # Split dataset to train and test
        splits = dataset.train_test_split(
            test_size=self.config.test_size, seed=42, shuffle=True
        )

        train_dataset = splits["train"]
        describe_dataset(dataset=train_dataset, split="train")

        test_dataset = splits["test"]
        describe_dataset(dataset=test_dataset, split="test")

        # Save datasets
        output_dir = "data"
        train_dataset.save_to_disk(f"{output_dir}/train")
        test_dataset.save_to_disk(f"{output_dir}/test")

        print(
            f"{Fore.GREEN}Preprocessing pipeline completed successfully!{Style.RESET_ALL}"
        )
