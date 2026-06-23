"""SignalForge exception hierarchy."""


class SignalForgeError(Exception):
    """Base exception for all SignalForge errors."""


class DataError(SignalForgeError):
    """Errors related to data loading, validation, or ingestion."""


class FeatureError(SignalForgeError):
    """Errors during feature engineering."""


class ModelError(SignalForgeError):
    """Errors during model training, prediction, or loading."""


class ConfigError(SignalForgeError):
    """Errors related to configuration validation."""


class BacktestError(SignalForgeError):
    """Errors during backtesting."""


class PaperTradingError(SignalForgeError):
    """Errors during paper trading lifecycle."""


class DatabaseError(SignalForgeError):
    """Errors related to database operations."""


class RegistryError(SignalForgeError):
    """Errors related to the model registry."""


class OptimizationError(SignalForgeError):
    """Errors during hyperparameter optimization."""


class DiscoveryError(SignalForgeError):
    """Errors during stock discovery."""


class ServerError(SignalForgeError):
    """Errors in the web server."""
