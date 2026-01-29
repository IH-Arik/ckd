"""
Configuration Management
=========================

Utilities for loading and managing configuration files.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}. Using default configuration.")
        return get_default_config()
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        print(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        print(f"Error loading config: {e}. Using default configuration.")
        return get_default_config()


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        print(f"Configuration saved to {config_path}")
        
    except Exception as e:
        print(f"Error saving config: {e}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'model': {
            'n_estimators': 100,
            'max_depth': 10,
            'learning_rate': 0.1,
            'max_iter': 500,
            'n_neighbors': 5,
            'hidden_layers': [100, 50]
        },
        'preprocessing': {
            'knn_neighbors': 5,
            'test_size': 0.2,
            'random_state': 42
        },
        'training': {
            'cv_folds': 5,
            'scoring': 'roc_auc',
            'n_jobs': -1
        },
        'data': {
            'target_column': 'ckd_status',
            'numeric_features': None,  # Will be auto-detected
            'categorical_features': None  # Will be auto-detected
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'preprocessing', 'training']
    
    for section in required_sections:
        if section not in config:
            print(f"Missing required config section: {section}")
            return False
    
    # Validate model parameters
    model_config = config['model']
    if 'n_estimators' in model_config and not isinstance(model_config['n_estimators'], int):
        print("n_estimators must be an integer")
        return False
    
    if 'max_depth' in model_config and not isinstance(model_config['max_depth'], int):
        print("max_depth must be an integer")
        return False
    
    return True


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
