from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

# Importing pipeline registers FeatureGenerator for joblib deserialization.
from .pipeline import FeatureGenerator  # noqa: F401


class LocalModelService:
    """Utility to lazily load and serve the serialized pipeline."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._pipeline = None

    def load(self):
        if self._pipeline is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el modelo entrenado en {self.model_path}. "
                    "Ejecutá el notebook tp3_modelado_vuelos.ipynb para generarlo."
                )
            self._pipeline = joblib.load(self.model_path)
        return self._pipeline

    def predict(self, payload: Dict[str, Any]) -> float:
        model = self.load()
        input_df = pd.DataFrame([payload])
        prediction = model.predict(input_df)[0]
        return float(prediction)


__all__ = ["LocalModelService"]

