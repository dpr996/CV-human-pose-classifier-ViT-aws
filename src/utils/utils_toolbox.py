from datasets import Dataset
from colorama import Fore, Style
from collections import Counter


def describe_dataset(dataset: Dataset, split: str) -> None:
    """
    Print a summary of a Hugging Face Dataset.

    Args:
        dataset (Dataset): The Hugging Face dataset to describe.
        split (str): The name of the split.
    """

    # Basic dataset info
    print(f"{Fore.CYAN}=== {split.upper()} DATASET SUMMARY ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Number of samples:{Style.RESET_ALL} {len(dataset)}")
    print(f"{Fore.YELLOW}Columns:{Style.RESET_ALL} {dataset.column_names}")

    # Show label distribution if applicable
    if "labels" in dataset.column_names:
        label_counts = Counter(dataset["labels"])
        print(f"{Fore.YELLOW}There are {len(label_counts)} classes {Style.RESET_ALL}")

        if hasattr(dataset.features["labels"], "names"):
            label_names = dataset.features["labels"].names
            label_counts_named = {label_names[i]: c for i, c in label_counts.items()}
            print(
                f"{Fore.YELLOW}Labels distribution:{Style.RESET_ALL} {label_counts_named}"
            )
        else:
            print(f"{Fore.YELLOW}Label distribution:{Style.RESET_ALL} {label_counts}")
