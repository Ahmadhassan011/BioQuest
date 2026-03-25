"""
Configuration management for BioQuest.

Features:
- Load/save JSON configurations
- Nested key access with dot notation
- Configuration validation against schema
- Environment variable substitution
- Default value fallback
- Type hints and comprehensive docstrings
"""

import json
import logging
import os
from typing import Dict, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


class Config:
    """
    Configuration manager for BioQuest.

    Features:
    - Load configurations from JSON files
    - Save configurations to JSON files
    - Nested key access using dot notation (e.g., 'optimization.max_iterations')
    - Environment variable substitution (e.g., '${HOME}/data')
    - Configuration validation against optional schema
    - Type-safe access with defaults
    - Pretty printing
    """

    # Schema for configuration validation
    SCHEMA: Dict[str, Dict[str, Union[str, int, float]]] = {
        "protein_sequence": {"type": "string", "required": True, "min_length": 5},
        "seeds": {"type": "list", "required": True, "min_items": 1},
        "objectives": {
            "type": "dict",
            "required": True,
            "allowed_keys": ["affinity", "toxicity", "qed", "sa", "diversity"],
            "sum_to": 1.0,  # Weights should sum to 1.0
        },
        "optimization": {
            "type": "dict",
            "fields": {
                "max_iterations": {"type": "int", "min": 1},
                "batch_size": {"type": "int", "min": 1},
                "patience": {"type": "int", "min": 1},
                "plateau_threshold": {"type": "float", "min": 0},
            },
        },
        "predictor": {
            "type": "dict",
            "fields": {
                "use_gpu": {"type": "bool"},
                "models_dir": {"type": "string"},
                "batch_predict": {"type": "bool"},
            },
        },
        "generation": {
            "type": "dict",
            "fields": {
                "vae_enabled": {"type": "bool"},
                "evolutionary_enabled": {"type": "bool"},
            },
        },
    }

    def __init__(self, config_file: Optional[str] = None, validate: bool = True):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to JSON configuration file
            validate: Whether to validate config against schema

        Raises:
            ConfigValidationError: If validation fails and validate=True
            FileNotFoundError: If config_file specified but doesn't exist
        """
        self.config = self._get_default_config()
        self.validate = validate

        if config_file:
            self.load(config_file)
            if validate:
                self.validate_config()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration with sensible defaults."""
        return {
            "protein_sequence": "",
            "seeds": [],
            "objectives": {
                "affinity": 0.4,
                "toxicity": 0.2,
                "qed": 0.2,
                "sa": 0.2,
            },
            "optimization": {
                "max_iterations": 100,
                "batch_size": 50,
                "patience": 20,
                "plateau_threshold": 0.001,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
            },
            "predictor": {
                "use_gpu": False,
                "models_dir": "trained_models",
                "batch_predict": True,
            },
            "generation": {
                "vae_enabled": True,
                "evolutionary_enabled": True,
            },
            "evaluation": {
                "use_pareto": True,
                "convergence_window": 10,
                "early_stopping": True,
            },
        }

    @staticmethod
    def _substitute_env_vars(value: Any) -> Any:
        """
        Recursively substitute environment variables in config values.

        Supports: ${VAR_NAME} or $VAR_NAME syntax

        Args:
            value: Config value (string, dict, list, or other)

        Returns:
            Value with environment variables substituted
        """
        if isinstance(value, str):
            # Substitute environment variables
            return os.path.expandvars(value)
        elif isinstance(value, dict):
            return {k: Config._substitute_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [Config._substitute_env_vars(v) for v in value]
        else:
            return value

    def load(self, config_file: str) -> None:
        """
        Load configuration from JSON file.

        Supports environment variable substitution in values.

        Args:
            config_file: Path to JSON configuration file

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        logger.info(f"Loading config from: {config_file}")

        with open(path, "r") as f:
            loaded_config = json.load(f)

        # Substitute environment variables
        loaded_config = self._substitute_env_vars(loaded_config)

        # Merge with defaults (new keys added, existing keys overridden)
        self.config = {**self.config, **loaded_config}

        logger.info(f"Config loaded with {len(self.config)} top-level keys")

    def save(self, config_file: str, pretty: bool = True) -> None:
        """
        Save configuration to JSON file.

        Args:
            config_file: Path to save JSON configuration
            pretty: Whether to pretty-print JSON (default: True)
        """
        path = Path(config_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to: {config_file}")

        with open(path, "w") as f:
            indent = 2 if pretty else None
            json.dump(self.config, f, indent=indent)

        logger.info("Config saved successfully")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation for nested keys.

        Examples:
            config.get('optimization.max_iterations')
            config.get('objectives.affinity', 0.4)

        Args:
            key: Configuration key (dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation for nested keys.

        Examples:
            config.set('optimization.max_iterations', 200)
            config.set('new_key', 'new_value')

        Args:
            key: Configuration key (dot notation for nested keys)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        # Navigate/create nested structure
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                raise ValueError(f"Cannot set nested value in non-dict: {k}")
            config = config[k]

        config[keys[-1]] = value
        logger.debug(f"Config set: {key} = {value}")

    def validate_config(self) -> None:
        """
        Validate configuration against schema.

        Raises:
            ConfigValidationError: If validation fails
        """
        logger.info("Validating configuration...")

        errors = []

        # Check protein sequence
        protein = self.config.get("protein_sequence", "")
        if not protein:
            errors.append("protein_sequence: Required field missing")
        elif len(protein) < 5:
            errors.append("protein_sequence: Must be at least 5 characters")

        # Check seeds
        seeds = self.config.get("seeds", [])
        if not seeds:
            errors.append("seeds: At least one seed molecule required")
        elif not isinstance(seeds, list):
            errors.append("seeds: Must be a list")

        # Check objectives
        objectives = self.config.get("objectives", {})
        if not objectives:
            errors.append("objectives: Must specify at least one objective")
        elif not isinstance(objectives, dict):
            errors.append("objectives: Must be a dictionary")
        else:
            total = sum(objectives.values())
            if abs(total - 1.0) > 0.01:  # Allow 1% tolerance
                errors.append(f"objectives: Weights must sum to 1.0 (got {total:.3f})")

        # Check optimization parameters
        optimization = self.config.get("optimization", {})
        if not isinstance(optimization, dict):
            errors.append("optimization: Must be a dictionary")
        else:
            max_iter = optimization.get("max_iterations", 100)
            if not isinstance(max_iter, int) or max_iter < 1:
                errors.append("optimization.max_iterations: Must be positive integer")

            batch_size = optimization.get("batch_size", 50)
            if not isinstance(batch_size, int) or batch_size < 1:
                errors.append("optimization.batch_size: Must be positive integer")

        if errors:
            error_msg = "\n".join(f"  • {e}" for e in errors)
            raise ConfigValidationError(
                f"Configuration validation failed:\n{error_msg}"
            )

        logger.info("✓ Configuration validated successfully")

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.

        Args:
            updates: Dictionary of {key: value} pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'optimization', 'objectives')

        Returns:
            Dictionary with section configuration
        """
        return self.config.get(section, {})

    def to_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary."""
        return dict(self.config)

    def __getitem__(self, key: str) -> Any:
        """Dict-like access: config['key']"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like setting: config['key'] = value"""
        self.config[key] = value

    def __repr__(self) -> str:
        """Pretty representation of config."""
        return f"Config({json.dumps(self.config, indent=2)})"

    def __str__(self) -> str:
        """String representation of config."""
        return json.dumps(self.config, indent=2)
