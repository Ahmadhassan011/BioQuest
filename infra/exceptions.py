"""
Custom exceptions for BioQuest.
"""


class BioQuestException(Exception):
    """Base exception for BioQuest."""


class DataError(BioQuestException):
    """Data loading or processing error."""


class DataProcessingError(DataError):
    """Error during data preprocessing or featurization."""


class CacheError(DataError):
    """Cache read/write error."""


class ModelError(BioQuestException):
    """Model-related error."""


class TrainingError(BioQuestException):
    """Training-related error."""


class ConfigurationError(BioQuestException):
    """Configuration error."""