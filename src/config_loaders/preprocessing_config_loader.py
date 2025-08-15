import json
from pydantic import BaseModel, Field
from typing import Optional


class PreprocessingConfig(BaseModel):
    huggingface_dataset_name: Optional[str] = Field(
        default="Bingsu/Human_Action_Recognition",
        description="Huggingface dataset name",
    )
    test_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Proportion of data used for testing (must be strictly between 0 and 1)",
    )
    validation_size: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Proportion of data used for validation (must be strictly between 0 and 1)",
    )
    output_dir: str = Field(default="data", description="Directory to save datasets")


def load_preprocessing_config(config_path: str) -> PreprocessingConfig:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return PreprocessingConfig(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
