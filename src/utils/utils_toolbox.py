from datasets import Dataset
from colorama import Fore, Style
from collections import Counter
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import os


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


def clean_checkpoints(train_dir: str) -> None:
    checkpoint_paths = Path(train_dir).glob("checkpoint-*")
    for path in checkpoint_paths:
        print(f"Removing old checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)

    print(Fore.MAGENTA + f"Old checkpoints in {train_dir} removed." + Style.RESET_ALL)


def plot_training_and_validation_curves(
    train_losses: list, val_losses: list, val_metrics: list, save_path: str
) -> None:

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # loss plot
    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses, label="Validation Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax1.legend()
    ax1.grid(True)

    # metrics plot
    ax2.plot(val_metrics, label="Validation Accuracy", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    # Save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(Fore.MAGENTA + f"Graph saved at {save_path}." + Style.RESET_ALL)
