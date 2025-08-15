import json
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    auto_image_processor_model: str = Field(
        ..., description="The model to use for image processing"
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
