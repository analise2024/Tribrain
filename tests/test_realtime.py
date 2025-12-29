import time
import pytest
from tribrain.core.realtime import RealTimeController, ResourceTimeoutError

def test_realtime_controller_basics():
    """Test basic functionality of real-time controller"""
    controller = RealTimeController(total_deadline=1.0)
    controller.start()
    
    # Should have full time remaining at start
    assert controller.get_remaining_time() <= 1.0
    assert controller.get_remaining_time() >= 0.9  # Allow for small timing variations
    
    # Sleep for 0.2 seconds
    time.sleep(0.2)
    assert controller.get_remaining_time() <= 0.8
    assert controller.get_remaining_time() >= 0.7

def test_time_budget_allocation():
    """Test time budget allocation across brains"""
    controller = RealTimeController(
        total_deadline=1.0,
        brain_weights=(0.2, 0.3, 0.5)
    )
    assert controller.get_time_budget('reactive') == 0.2
    assert controller.get_time_budget('contextual') == 0.3
    assert controller.get_time_budget('rational') == 0.5

def test_deadline_enforcement():
    """Test that deadlines are enforced properly"""
    controller = RealTimeController(total_deadline=0.5)
    controller.start()
    
    # This function should time out
    def slow_function():
        time.sleep(0.6)
        return "done"
    
    with pytest.raises(ResourceTimeoutError):
        controller.process_with_deadline('reactive', slow_function)
    
    # Verify timing was tracked
    timings = controller.get_timing_report()
    assert timings['reactive']['used'] >= 0.5
    assert timings['reactive']['percentage'] >= 100

def test_dynamic_time_reallocation():
    """Test that time is reallocated when earlier brains finish early"""
    controller = RealTimeController(
        total_deadline=1.0,
        brain_weights=(0.2, 0.3, 0.5)
    )
    controller.start()
    
    # First brain finishes early
    controller.track_time('reactive', 0.1)  # Used only half its budget
    
    # Second brain should now have more time available
    contextual_budget = controller.get_time_budget('contextual')
    assert contextual_budget > 0.3  # Should have inherited some time
    assert contextual_budget <= 0.4  # But not more than total remaining
    
    # Verify rational brain also gets more time
    rational_budget = controller.get_time_budget('rational')
    assert rational_budget > 0.5