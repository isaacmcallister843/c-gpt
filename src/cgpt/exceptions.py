
class CheckpointCorruptionError(Exception):
    """Raised when a checkpoint file is found to be corrupted or unreadable."""
    pass

class IllegalMoveError(Exception):
    """Raised when the model generates an illegal move in chess."""
    pass

class ConfigurationError(Exception):
    """Raised when there is an issue with the configuration file or parameters."""
    pass
