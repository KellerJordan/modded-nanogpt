import os
import pickle 

def serialize_matrix_params(model, filepath):
    """
    Fast serialization of matrix parameters using pickle, including parameter names.
    
    Args:
        model: The GPT model (will be unwrapped if DDP)
        filepath: Path to save the pickle file
    """
    # Unwrap DDP if needed
    if hasattr(model, 'module'):
        model = model.module
    
    # Get all transformer matrix parameters with their names in one pass
    named_params = [
        (name.replace('module.', '').replace('transformer.', '').replace('weight', '').strip('.'),
         param.detach().cpu().numpy(),
         param.shape)
        for name, param in model.named_parameters()
        if param.ndim == 2 and 'transformer.h' in name
    ]
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(named_params, f)