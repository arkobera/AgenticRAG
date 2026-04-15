"""
Configuration loader for RAG pipeline.
Loads parameters from config.yaml for centralized management.
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager for RAG pipeline"""
    
    _instance = None
    _config_data = None
    
    def __new__(cls):
        """Singleton pattern - only one config instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls) -> None:
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                "Please ensure config.yaml exists in the project root directory."
            )
        
        with open(config_path, 'r') as f:
            cls._config_data = yaml.safe_load(f)
        
        print(f"✓ Configuration loaded from {config_path}")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Examples:
            Config.get("document_processing.chunk_size")
            Config.get("llm.max_new_tokens")
            Config.get("retriever.dense_weight")
        
        Args:
            key: Configuration key using dot notation (section.subsection.key)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if cls._config_data is None:
            instance = cls()
        
        keys = key.split('.')
        value = cls._config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            if default is not None:
                return default
            raise KeyError(f"Configuration key not found: {key}")
    
    @classmethod
    def get_section(cls, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Examples:
            Config.get_section("document_processing")
            Config.get_section("llm")
            
        Args:
            section: Section name
            
        Returns:
            Dictionary with section configuration
        """
        if cls._config_data is None:
            instance = cls()
        
        if section not in cls._config_data:
            raise KeyError(f"Configuration section not found: {section}")
        
        return cls._config_data[section]
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        if cls._config_data is None:
            instance = cls()
        return cls._config_data
    
    @classmethod
    def reload(cls) -> None:
        """Reload configuration from file"""
        cls._config_data = None
        cls._instance = None
        instance = cls()


# Convenience function to avoid verbosity
def get_config(key: str, default: Any = None) -> Any:
    """Convenience function to get config value"""
    return Config.get(key, default)
