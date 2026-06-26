class VisualizationError(RuntimeError):
    """Raised when dashboard state cannot be persisted or rendered."""


class TrainingStopRequested(RuntimeError):
    """Raised by adapters that need to stop training at a safe boundary."""
