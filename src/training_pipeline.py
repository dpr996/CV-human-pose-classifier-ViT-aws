from src.config_loaders.training_config_loader import TrainingConfig
from src.base_pipeline import BasePipeline
from colorama import Fore, Style


class TrainingPipeline(BasePipeline):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def run(self) -> None:

        print(f"{Fore.GREEN}Starting training pipeline...{Style.RESET_ALL}")

        print(f"{Fore.GREEN}Training pipeline completed successfully!{Style.RESET_ALL}")
