import json
from pydantic import BaseModel, Field
from typing import Optional


class TrainingConfig(BaseModel):
    model_name: str = Field(
        ..., description="The model to use for image classification"
    )
    input_dir: str = Field(
        default="data", description="Directory to load train and validation datasets"
    )
    enable_gpu: Optional[bool] = Field(
        default=False, description="Whether to use GPU if available"
    )


def load_training_config(config_path: str) -> TrainingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return TrainingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
