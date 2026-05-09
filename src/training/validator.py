"""Model validation and sanity checks for inputs, outputs, and configs."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates models and their predictions."""

    @staticmethod
    def validate_input_shape(
        data: np.ndarray,
        expected_shape: Tuple[int, ...],
        name: str = "input",
    ) -> bool:
        if data.shape != expected_shape:
            raise ValueError(
                f"{name} shape mismatch. Expected {expected_shape}, got {data.shape}"
            )
        return True

    @staticmethod
    def validate_prediction_range(
        predictions: np.ndarray,
        min_val: float = 0.0,
        max_val: float = 1.0,
        name: str = "prediction",
    ) -> Tuple[bool, List[int]]:
        out_of_range = np.where(
            (predictions < min_val) | (predictions > max_val)
        )[0].tolist()

        is_valid = len(out_of_range) == 0

        if not is_valid:
            logger.warning(
                f"{name} contains {len(out_of_range)} values out of range "
                f"[{min_val}, {max_val}]"
            )

        return is_valid, out_of_range

    @staticmethod
    def validate_no_nan(values: np.ndarray, name: str = "values") -> bool:
        if np.isnan(values).any():
            nan_count = np.isnan(values).sum()
            raise ValueError(f"{name} contains {nan_count} NaN values")

        return True

    @staticmethod
    def validate_model_config(config: Dict, required_keys: List[str]) -> bool:
        missing_keys = [key for key in required_keys if key not in config]

        if missing_keys:
            raise ValueError(
                f"Model config missing required keys: {missing_keys}"
            )

        return True

    @staticmethod
    def check_gradient_health(gradients: Dict[str, Optional[np.ndarray]]) -> Dict[str, bool]:
        health_report = {}

        for name, grad in gradients.items():
            if grad is None:
                health_report[name] = False
                logger.warning(f"Gradient '{name}' is None")
                continue

            has_nan = np.isnan(grad).any()
            has_inf = np.isinf(grad).any()
            is_zero = np.allclose(grad, 0)

            is_healthy = not (has_nan or has_inf or is_zero)
            health_report[name] = is_healthy

            if not is_healthy:
                if has_nan:
                    logger.warning(f"Gradient '{name}' contains NaN")
                if has_inf:
                    logger.warning(f"Gradient '{name}' contains Inf")
                if is_zero:
                    logger.warning(f"Gradient '{name}' is all zeros")

        return health_report
