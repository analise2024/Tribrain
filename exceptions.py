"""
Custom exceptions for Tribrain system
"""
class TribrainError(Exception):
    """Base exception for Tribrain application"""
    pass

class SecurityError(TribrainError):
    """Raised when security validation fails"""
    pass

class ConfigurationError(TribrainError):
    """Raised when configuration is invalid"""
    pass

class BrainProcessingError(TribrainError):
    """Raised during brain processing failures"""
    def __init__(self, brain_name: str, message: str, original_error: Exception = None):
        self.brain_name = brain_name
        super().__init__(f"{brain_name} brain error: {message}")
        self.original_error = original_error

class ResourceTimeoutError(TribrainError):
    """Raised when a resource operation times out"""
    def __init__(self, resource: str, timeout: float):
        super().__init__(f"{resource} operation timed out after {timeout} seconds")
        self.resource = resource
        self.timeout = timeout

class ConsensusFailureError(TribrainError):
    """Raised when brains cannot reach consensus"""
    def __init__(self, disagreement_score: float, max_disagreement: float):
        super().__init__(
            f"Brains failed to reach consensus (disagreement: {disagreement_score:.2f} > {max_disagreement:.2f})"
        )
        self.disagreement_score = disagreement_score
        self.max_disagreement = max_disagreement