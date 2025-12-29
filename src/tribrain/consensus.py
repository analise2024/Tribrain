"""
Safety-first consensus engine for Tribrain system
"""
from typing import List, Dict, Tuple
import numpy as np
from .exceptions import ConsensusFailureError

def calculate_disagreement(brain_outputs: List[Dict]) -> float:
    """
    Calculate disagreement score between brains (0-1, 1=highest disagreement)
    Uses safety-critical parameters with higher weighting
    """
    # Check for critical action type disagreement (most important)
    action_types = [out.get('action_type', 'refine') for out in brain_outputs]
    if len(set(action_types)) > 1:
        return 1.0
    
    # Calculate parameter disagreement with safety weighting
    disagreements = []
    
    # Safety parameters get higher disagreement weight
    safety_params = ['safety_margin', 'velocity', 'force', 'trajectory']
    
    # Collect all parameters across brains
    all_params = set()
    for out in brain_outputs:
        if 'params' in out:
            all_params.update(out['params'].keys())
    
    for param in all_params:
        values = []
        for out in brain_outputs:
            if 'params' in out and param in out['params']:
                values.append(out['params'][param])
        
        if len(values) >= 2:  # Need at least 2 values to calculate disagreement
            # Higher weight for safety parameters
            weight = 2.0 if param in safety_params else 1.0
            
            # Calculate normalized disagreement (0-1)
            min_val, max_val = min(values), max(values)
            if min_val == max_val:
                param_disagreement = 0.0
            else:
                # Normalize to 0-1 range
                param_disagreement = (max_val - min_val) / (max_val + 1e-8)
            
            disagreements.append(weight * param_disagreement)
    
    return np.mean(disagreements) if disagreements else 0.0

def generate_safe_consensus(brain_outputs: List[Dict],
                           timing_data: Dict) -> Dict:
    """
    Generate consensus decision with safety prioritization
    
    Args:
        brain_outputs: Outputs from [reactive, contextual, rational] brains
        timing_data: Timing information for each brain
        
    Returns:
        Safe consensus decision with appropriate confidence
    """
    # 1. EMERGENCY CHECK: If reactive brain detected critical issue, prioritize safety
    if brain_outputs[0].get('emergency_stop', False):
        return _generate_emergency_response(
            brain_outputs[0],
            timing_data
        )
    
    # 2. Calculate disagreement score
    disagreement = calculate_disagreement(brain_outputs)
    
    # 3. High uncertainty protocol
    if disagreement > 0.4 or brain_outputs[0].get('uncertainty', 0) > 0.7:
        return _handle_high_uncertainty(
            brain_outputs,
            disagreement,
            timing_data
        )
    
    # 4. Normal consensus with safety bias
    return _generate_weighted_consensus(
        brain_outputs,
        disagreement,
        timing_data
    )

def _generate_emergency_response(reactive_output: Dict, timing_data: Dict) -> Dict:
    """Generate response for emergency stop situations"""
    return {
        'action': 'SAFE_HOLD',
        'action_type': 'emergency',
        'params': {
            'safety_margin': 0.5,
            'recovery_steps': reactive_output.get('recovery_steps', [])
        },
        'confidence': 0.95,
        'reason': f"Emergency stop: {reactive_output.get('emergency_reason', 'safety risk')}",
        'source': 'reactive',
        'disagreement': 0.0,  # Not applicable in emergency
        'timing': timing_data
    }

def _handle_high_uncertainty(brain_outputs: List[Dict],
                           disagreement: float,
                           timing_data: Dict) -> Dict:
    """Handle situations with high uncertainty or disagreement"""
    # Use reactive brain's output as safest option
    reactive = brain_outputs[0]
    
    # Add safety margin to any motion commands
    safety_margin = 0.3  # Base safety margin
    if 'params' in reactive and 'safety_margin' in reactive['params']:
        safety_margin = max(safety_margin, reactive['params']['safety_margin'] + 0.1)
    
    return {
        'action': reactive.get('refined_prompt', brain_outputs[2].get('refined_prompt', '')),
        'action_type': reactive.get('action_type', 'refine'),
        'params': {
            **reactive.get('params', {}),
            'safety_margin': safety_margin
        },
        'confidence': min(0.6, reactive.get('confidence', 0.5)),
        'reason': f'high uncertainty (disagreement: {disagreement:.2f})',
        'source': 'reactive',
        'disagreement': disagreement,
        'timing': timing_data
    }

def _generate_weighted_consensus(brain_outputs: List[Dict],
                               disagreement: float,
                               timing_data: Dict) -> Dict:
    """Generate consensus with safety-biased weighting"""
    # Base weights: [reactive, contextual, rational]
    base_weights = [0.15, 0.35, 0.5]
    
    # Adjust weights based on situation
    if brain_outputs[2].get('task_type') == 'human_interaction':
        base_weights = [0.1, 0.4, 0.5]  # More weight on social context
    elif brain_outputs[2].get('task_type') == 'precision_manipulation':
        base_weights = [0.2, 0.25, 0.55]  # More weight on precision planning
    
    # Safety boost for high-risk situations
    if brain_outputs[0].get('risk_level', 0) > 0.4:
        base_weights[0] += 0.15  # Boost reactive brain
        base_weights[2] = max(0.2, base_weights[2] - 0.15)  # Reduce rational brain
    
    # Normalize weights
    total = sum(base_weights)
    weights = [w/total for w in base_weights]
    
    # Calculate weighted confidence (safety parameters weighted higher)
    confidence = 0
    safety_params = ['safety_margin', 'velocity', 'force']
    
    for i, (brain, weight) in enumerate(zip(brain_outputs, weights)):
        brain_conf = brain.get('confidence', 0.5)
        
        # Boost confidence if safety parameters are present
        if 'params' in brain:
            safety_present = any(param in brain['params'] for param in safety_params)
            if safety_present:
                brain_conf = min(1.0, brain_conf * 1.2)
        
        confidence += brain_conf * weight
    
    # Get the rational brain's output as primary (safest when consensus is reached)
    rational = brain_outputs[2]
    action = rational.get('refined_prompt', '')
    
    # Apply safety margins to physical actions
    params = rational.get('params', {}).copy()
    if 'motion' in rational.get('action_type', ''):
        # Calculate dynamic safety margin based on uncertainty
        base_margin = params.get('safety_margin', 0.1)
        uncertainty_factor = min(1.0, disagreement * 2)
        params['safety_margin'] = base_margin + (0.2 * uncertainty_factor)
    
    return {
        'action': action,
        'action_type': rational.get('action_type', 'refine'),
        'params': params,
        'confidence': min(0.95, confidence),  # Cap at 95% confidence
        'reason': 'consensus reached',
        'source': 'rational',
        'disagreement': disagreement,
        'timing': timing_data,
        'weights': dict(zip(['reactive', 'contextual', 'rational'], weights))
    }