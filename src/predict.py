"""Model inference module for house price estimation."""

import os
from typing import Dict, Any, Union
import joblib
import numpy as np
import pandas as pd


def load_model_pipeline(model_path: str = "models/house_price_pipeline.joblib") -> Any:
    """Load the serialized scikit-learn / XGBoost pipeline artifact.

    Args:
        model_path: Path to the .joblib artifact

    Returns:
        Loaded scikit-learn Pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Please run train.py first."
        )
    return joblib.load(model_path)


def predict_price(
    features: Union[Dict[str, Any], pd.DataFrame],
    pipeline: Any = None,
    model_path: str = "models/house_price_pipeline.joblib",
) -> float:
    """Predict sale price for a single house or a batch of properties.

    Args:
        features: Dictionary of feature key-values or a pandas DataFrame
        pipeline: Pre-loaded pipeline instance (optional)
        model_path: Path to pipeline artifact if pipeline is not pre-loaded

    Returns:
        Predicted house sale price in USD
    """
    if pipeline is None:
        pipeline = load_model_pipeline(model_path)

    if isinstance(features, dict):
        df_input = pd.DataFrame([features])
    else:
        df_input = features.copy()

    # Drop ID columns if present
    for col in ["Order", "PID", "SalePrice"]:
        if col in df_input.columns:
            df_input = df_input.drop(columns=[col])

    prediction = pipeline.predict(df_input)
    return float(np.round(prediction[0] if len(prediction) == 1 else prediction, 2))
