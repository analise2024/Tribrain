"""
Real-time processing constraints for Tribrain system
"""
import time
from typing import Callable, Any, Dict, List, Tuple
from .exceptions import ResourceTimeoutError

class RealTimeController:
    """
    Manages real-time constraints across the three brains
    Ensures each brain completes within its allocated time budget
    """
    def __init__(self,
                 total_deadline: float = 1.0,
                 brain_weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)):
        """
        Initialize real-time controller
        
        Args:
            total_deadline: Total time budget for all brains (seconds)
            brain_weights: Time allocation weights for [reactive, contextual, rational]
        """
        self.total_deadline = total_deadline
        self.brain_weights = brain_weights
        self.start_time = None
        self.timings = {
            'reactive': {'allocated': total_deadline * brain_weights[0], 'used': 0},
            'contextual': {'allocated': total_deadline * brain_weights[1], 'used': 0},
            'rational': {'allocated': total_deadline * brain_weights[2], 'used': 0}
        }
    
    def start(self):
        """Start the timing process"""
        self.start_time = time.time()
    
    def get_remaining_time(self) -> float:
        """Get remaining time in the current deadline"""
        if self.start_time is None:
            raise RuntimeError("RealTimeController not started")
        elapsed = time.time() - self.start_time
        return max(0, self.total_deadline - elapsed)
    
    def get_time_budget(self, brain_name: str) -> float:
        """Get time budget for a specific brain"""
        if brain_name not in self.timings:
            raise ValueError(f"Unknown brain: {brain_name}")
        
        # Calculate dynamic budget based on remaining time
        remaining = self.get_remaining_time()
        base_budget = self.timings[brain_name]['allocated']
        
        # Don't allocate more than remaining time
        return min(base_budget, remaining)
    
    def track_time(self, brain_name: str, time_used: float):
        """Track time used by a brain"""
        if brain_name in self.timings:
            self.timings[brain_name]['used'] = time_used
    
    def process_with_deadline(self,
                              brain_name: str,
                              func: Callable,
                              *args,
                              **kwargs) -> Any:
        """
        Process a function with a deadline for a specific brain
        
        Args:
            brain_name: Name of the brain ('reactive', 'contextual', or 'rational')
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function execution
            
        Raises:
            ResourceTimeoutError: If function doesn't complete in time
        """
        if self.start_time is None:
            self.start()
        
        deadline = self.get_time_budget(brain_name)
        start = time.time()
        
        try:
            # Execute with the calculated deadline
            result = self._execute_with_timeout(func, deadline, *args, **kwargs)
            time_used = time.time() - start
            self.track_time(brain_name, time_used)
            return result
        except ResourceTimeoutError:
            time_used = time.time() - start
            self.track_time(brain_name, time_used)
            raise
    
    def _execute_with_timeout(self, func, timeout, *args, **kwargs):
        """Execute function with timeout protection"""
        import threading
        
        class TimeoutException(Exception):
            pass
        
        result = [TimeoutException("Operation timed out")]
        
        def target():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                result[0] = e
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise ResourceTimeoutError(f"{func.__name__}", timeout)
        
        if isinstance(result[0], Exception):
            raise result[0]
        
        return result[0]
    
    def get_timing_report(self) -> Dict[str, Dict]:
        """Get detailed timing report for all brains"""
        return {
            name: {
                'allocated': timing['allocated'],
                'used': timing['used'],
                'percentage': (timing['used'] / timing['allocated'] * 100) if timing['allocated'] > 0 else 0
            }
            for name, timing in self.timings.items()
        }